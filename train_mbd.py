import yaml
import random
import torch
import torchmetrics
import time
import numpy as np
import torch.distributed as dist
from datetime import datetime
from argparse import ArgumentParser
from data.flow_viz import trend_plus_vis
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from model.MBD import MBD
from model.utils import AverageMeter
from os.path import join
from logger import Logger
import lpips

import sys
import traceback

def excepthook(type, value, tb):
    traceback.print_exception(type, value, tb)
    sys.exit(1)

sys.excepthook = excepthook

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_gaussian_noise(images, noise_prob=0.5, sigma_range=(5, 25)):
    """
    Add Gaussian noise to images with given probability

    Args:
        images: torch.Tensor (B, C, H, W), values in [0, 1] range
        noise_prob: probability of adding noise (0.5 = 50%)
        sigma_range: tuple of (min_sigma, max_sigma) for noise level

    Returns:
        noisy_images: torch.Tensor with noise added
        noise_levels: torch.Tensor (B, 1) actual sigma values used (0 if no noise)
    """
    B = images.size(0)
    device = images.device

    # Decide which samples get noise
    noise_mask = torch.rand(B, device=device) < noise_prob  # (B,)

    # Generate random sigma values for each sample
    sigma_min, sigma_max = sigma_range
    sigma_values = torch.rand(B, device=device) * (sigma_max - sigma_min) + sigma_min  # (B,)

    # Zero out sigma for samples that don't get noise
    sigma_values = sigma_values * noise_mask.float()  # (B,)

    # Add noise
    noisy_images = images.clone()
    for i in range(B):
        if noise_mask[i]:
            noise = torch.randn_like(images[i]) * (sigma_values[i] / 255.0)
            noisy_images[i] = images[i] + noise
            noisy_images[i] = torch.clamp(noisy_images[i], 0, 1)

    return noisy_images, sigma_values.unsqueeze(1)  # (B, 1)


def train(local_rank, configs, log_dir):
    # Preparation and backup
    device = torch.device("cuda", args.local_rank)
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        writer = SummaryWriter(log_dir)
        configs_bp = join(log_dir, 'cfg.yaml')
        with open(configs_bp, 'w') as f:
            yaml.dump(configs, f)
    else:
        writer = None
    step = 0
    num_eval = 0

    # model init
    model = MBD(local_rank=local_rank, configs=configs)

    # dataset init
    dataset_args = configs['dataset_args']
    train_dataset = BDDataset(set_type='train', **dataset_args)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=configs['train_batch_size'],
                              num_workers=configs['num_workers'],
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler)
    valid_dataset = BDDataset(set_type='valid', **dataset_args)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=configs['valid_batch_size'],
                              num_workers=configs['num_workers'],
                              pin_memory=True)

    # Get noise augmentation settings from config (with defaults)
    noise_aug_prob = configs.get('noise_aug_prob', 0.45)  # 45% of batches get noise
    noise_sigma_range = tuple(configs.get('noise_sigma_range', [5, 25]))

    # training looping
    step_per_epoch = len(train_loader)
    time_stamp = time.time()
    for epoch in range(configs['epoch']):
        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch)
        for i, tensor in enumerate(train_loader):
            # Record time after loading data
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            # Move to device
            tensor['inp'] = tensor['inp'].to(device)  # (b, 1, 3, h, w)
            tensor['gt'] = tensor['gt'].to(device)  # (b, num_gts, 3, h, w)
            # Handle trend - create dummy if not present (GoPro doesn't have trend)
            if 'trend' in tensor and tensor['trend'] is not None:
                tensor['trend'] = tensor['trend'].to(device)
            else:
                b, _, c, h, w = tensor['inp'].shape
                tensor['trend'] = torch.zeros(b, 1, 2, h, w, device=device)

            # === NEW: Add noise augmentation ===
            b = tensor['inp'].size(0)
            if random.random() < noise_aug_prob:
                # Squeeze from (b, 1, 3, h, w) to (b, 3, h, w) for noise function
                # Note: inp is 0-255 range, need to normalize first
                inp_squeezed = tensor['inp'].squeeze(1) / 255.0  # Normalize to [0, 1]
                noisy_inp, noise_levels = add_gaussian_noise(
                    inp_squeezed,
                    noise_prob = 1.0,  # All samples in this batch get noise
                    sigma_range = noise_sigma_range
                )
                tensor['inp'] = (noisy_inp * 255.0).unsqueeze(1)  # Back to (b, 1, 3, h, w) and 0-255
                tensor['noise_level'] = noise_levels.to(device)
            else:
                # No noise - set noise level to 0
                tensor['noise_level'] = torch.zeros(b, 1, device=device)

            # Update model
            gt_flow_ratio = 1 - epoch / (configs['epoch'] - 1)
            hybrid_flag = np.random.choice(np.arange(0, 2), p=[1 - gt_flow_ratio, gt_flow_ratio])
            out_tensor = model.update(inp_tensor=tensor, hybrid_flag=hybrid_flag, training=True)
            loss = out_tensor['loss']

            # Record time after updating model
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            # Print training info
            if step % 100 == 0:
                if rank == 0:
                    writer.add_scalar('learning_rate', model.get_lr(), step)
                    writer.add_scalar('train/loss', loss.item(), step)

                    # NEW: Log noise-related metrics
                    avg_noise = tensor['noise_level'].mean().item()
                    writer.add_scalar('train/noise_level', avg_noise, step)
                    if 'noise_est_loss' in out_tensor:
                        writer.add_scalar('train/noise_est_loss', out_tensor['noise_est_loss'].item(), step)

                    msg = 'epoch: {:>3}, batch: [{:>5}/{:>5}], time: {:.2f} + {:.2f} sec, loss: {:.5f}, noise: {:.1f}'
                    msg = msg.format(epoch + 1,
                                     i + 1,
                                     step_per_epoch,
                                     data_time_interval,
                                     train_time_interval,
                                     loss.item(),
                                     avg_noise)
                    logger(msg, prefix='[train]')

            if (rank == 0) and (step % 500 == 0):
                inp_img = out_tensor['inp_img']  # inp_img shape (b, c, h, w)
                trend_img = out_tensor['trend_img']  # trend_img shape (b, 2, h, w)
                pred_imgs = out_tensor['pred_imgs']  # pred_imgs shape (b, num_gts, 3, h, w)
                gt_imgs = out_tensor['gt_imgs']  # gt_imgs shape (b, num_gts, 3, h, w)

                # Prepare recorded results
                inp_img = inp_img.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)

                trend_img = trend_img.permute(0, 2, 3, 1).cpu().detach().numpy()
                trend_img_rgb = []
                for item in trend_img:
                    trend_img_rgb.append(trend_plus_vis(item))

                b, num_gts, c, h, w = pred_imgs.shape
                pred_imgs = pred_imgs.permute(0, 3, 1, 4, 2).reshape(b, h, num_gts * w, c)
                pred_imgs = pred_imgs.cpu().detach().numpy().astype(np.uint8)
                gt_imgs = gt_imgs.permute(0, 3, 1, 4, 2).reshape(b, h, num_gts * w, c)
                gt_imgs = gt_imgs.cpu().detach().numpy().astype(np.uint8)

                # Record each sample results in the batch
                for j in range(b):
                    cat_pred_imgs = np.concatenate([inp_img[j], trend_img_rgb[j], pred_imgs[j]], axis=1)
                    cat_gt_imgs = np.concatenate([inp_img[j], trend_img_rgb[j], gt_imgs[j]], axis=1)
                    cat_imgs = np.concatenate([cat_gt_imgs, cat_pred_imgs], axis=0)
                    writer.add_image('train/imgs_results_{}'.format(j), cat_imgs, step, dataformats='HWC')

            # Ending of a batch
            step += 1

        # Ending of an epoch
        num_eval += 1
        evaluate(model, valid_loader, num_eval, local_rank, writer)
        if rank == 0:
            model.save_model(log_dir)
            if num_eval % 5 == 0:
                model.save_model(log_dir, num_eval)
        model.scheduler_step()
        dist.barrier()


@torch.no_grad()
def evaluate(model, valid_loader, num_eval, local_rank, writer):
    # Preparation
    torch.cuda.empty_cache()
    device = torch.device("cuda", local_rank)
    loss_meter_clean = AverageMeter()
    loss_meter_noisy = AverageMeter()
    psnr_meter_clean = AverageMeter()
    ssim_meter_clean = AverageMeter()
    lpips_meter_clean = AverageMeter()
    psnr_meter_noisy = AverageMeter()
    ssim_meter_noisy = AverageMeter()
    lpips_meter_noisy = AverageMeter()

    # NEW: Add meter for noise estimation error
    noise_est_meter = AverageMeter()

    time_stamp = time.time()

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # One epoch validation
    random_idx = random.randint(0, len(valid_loader) - 1)
    for i, tensor in enumerate(valid_loader):
        tensor['inp'] = tensor['inp'].to(device)
        tensor['gt'] = tensor['gt'].to(device)
        if 'trend' in tensor and tensor['trend'] is not None:
            tensor['trend'] = tensor['trend'].to(device)
        else:
            b, _, c, h, w = tensor['inp'].shape
            tensor['trend'] = torch.zeros(b, 1, 2, h, w, device=device)

        b = tensor['inp'].size(0)

        # === Test 1: Clean images ===
        tensor['noise_level'] = torch.zeros(b, 1, device=device)
        out_tensor_clean = model.update(inp_tensor=tensor, training=False)
        pred_imgs_clean = out_tensor_clean['pred_imgs']
        gt_imgs = out_tensor_clean['gt_imgs']
        loss_clean = out_tensor_clean['loss']

        # Metrics for clean
        _, num_gts, c, h, w = pred_imgs_clean.shape
        psnr_clean = torchmetrics.functional.psnr(
            pred_imgs_clean.reshape(num_gts * b, c, h, w),
            gt_imgs.reshape(num_gts * b, c, h, w)
        )
        ssim_clean = torchmetrics.functional.ssim(
            pred_imgs_clean.reshape(num_gts * b, c, h, w),
            gt_imgs.reshape(num_gts * b, c, h, w)
        )

        # LPIPS for clean
        pred_normalized = (pred_imgs_clean.reshape(num_gts * b, c, h, w) / 255.0) * 2 - 1
        gt_normalized = (gt_imgs.reshape(num_gts * b, c, h, w) / 255.0) * 2 - 1
        lpips_clean = lpips_model(pred_normalized, gt_normalized).mean()

        loss_meter_clean.update(loss_clean.item(), b)
        psnr_meter_clean.update(psnr_clean, num_gts * b)
        ssim_meter_clean.update(ssim_clean, num_gts * b)
        lpips_meter_clean.update(lpips_clean.item(), num_gts * b)

        # === Test 2: Noisy images (fixed sigma=15) ===
        inp_squeezed = tensor['inp'].squeeze(1) / 255.0  # Normalize to [0, 1]
        noisy_inp, noise_levels = add_gaussian_noise(
            inp_squeezed,
            noise_prob=1.0,
            sigma_range=(15, 15)  # Fixed sigma for consistent evaluation
        )
        tensor_noisy = {
            'inp': (noisy_inp * 255.0).unsqueeze(1),
            'gt': tensor['gt'],
            'trend': tensor['trend'],
            'noise_level': noise_levels.to(device)
        }

        out_tensor_noisy = model.update(inp_tensor=tensor_noisy, training=False)
        pred_imgs_noisy = out_tensor_noisy['pred_imgs']
        loss_noisy = out_tensor_noisy['loss']

        # NEW: Get estimated noise level from model output
        if 'noise_level_est' in out_tensor_noisy:
            noise_est = out_tensor_noisy['noise_level_est']  # (b, 1)
            noise_gt = noise_levels  # (b, 1)
            noise_error = torch.abs(noise_est - noise_gt).mean()
            noise_est_meter.update(noise_error.item(), b)

        # Metrics for noisy
        psnr_noisy = torchmetrics.functional.psnr(
            pred_imgs_noisy.reshape(num_gts * b, c, h, w),
            gt_imgs.reshape(num_gts * b, c, h, w)
        )
        ssim_noisy = torchmetrics.functional.ssim(
            pred_imgs_noisy.reshape(num_gts * b, c, h, w),
            gt_imgs.reshape(num_gts * b, c, h, w)
        )

        # LPIPS for noisy
        pred_noisy_normalized = (pred_imgs_noisy.reshape(num_gts * b, c, h, w) / 255.0) * 2 - 1
        lpips_noisy = lpips_model(pred_noisy_normalized, gt_normalized).mean()

        loss_meter_noisy.update(loss_noisy.item(), b)
        psnr_meter_noisy.update(psnr_noisy, num_gts * b)
        ssim_meter_noisy.update(ssim_noisy, num_gts * b)
        lpips_meter_noisy.update(lpips_noisy.item(), num_gts * b)

        # Record image results (from clean version)
        if rank == 0 and i == random_idx:
            inp_img = out_tensor_clean['inp_img']
            inp_img = inp_img.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)

            trend_img = out_tensor_clean['trend_img']
            trend_img = trend_img.permute(0, 2, 3, 1).cpu().detach().numpy()
            trend_img_rgb = []
            for item in trend_img:
                trend_img_rgb.append(trend_plus_vis(item))

            pred_imgs_vis = pred_imgs_clean.permute(0, 3, 1, 4, 2).reshape(b, h, num_gts * w, c)
            pred_imgs_vis = pred_imgs_vis.cpu().numpy().astype(np.uint8)
            gt_imgs_vis = gt_imgs.permute(0, 3, 1, 4, 2).reshape(b, h, num_gts * w, c)
            gt_imgs_vis = gt_imgs_vis.cpu().numpy().astype(np.uint8)

            for j in range(b):
                cat_pred_imgs = np.concatenate([inp_img[j], trend_img_rgb[j], pred_imgs_vis[j]], axis=1)
                cat_gt_imgs = np.concatenate([inp_img[j], trend_img_rgb[j], gt_imgs_vis[j]], axis=1)
                cat_imgs = np.concatenate([cat_gt_imgs, cat_pred_imgs], axis=0)
                writer.add_image('valid/imgs_results_{}'.format(j), cat_imgs, num_eval, dataformats='HWC')

    # Ending of validation
    eval_time_interval = time.time() - time_stamp
    if rank == 0:
        writer.add_scalar('valid/loss_clean', loss_meter_clean.avg, num_eval)
        writer.add_scalar('valid/loss_noisy', loss_meter_noisy.avg, num_eval)
        writer.add_scalar('valid/psnr_clean', psnr_meter_clean.avg, num_eval)
        writer.add_scalar('valid/psnr_noisy', psnr_meter_noisy.avg, num_eval)
        writer.add_scalar('valid/ssim_clean', ssim_meter_clean.avg, num_eval)
        writer.add_scalar('valid/ssim_noisy', ssim_meter_noisy.avg, num_eval)
        writer.add_scalar('valid/lpips_clean', lpips_meter_clean.avg, num_eval)
        writer.add_scalar('valid/lpips_noisy', lpips_meter_noisy.avg, num_eval)

        # NEW: Log noise estimation error
        writer.add_scalar('valid/noise_est_error', noise_est_meter.avg, num_eval)

        # NEW: Updated message with noise estimation
        msg = 'eval time: {:.1f}s | clean (noise=0): loss={:.5f}, psnr={:.2f}, ssim={:.4f}, lpips={:.4f} | noisy (noise=15): loss={:.5f}, psnr={:.2f}, ssim={:.4f}, lpips={:.4f}, noise_est_error={:.2f}'
        msg = msg.format(
            eval_time_interval,
            loss_meter_clean.avg, psnr_meter_clean.avg, ssim_meter_clean.avg, lpips_meter_clean.avg,
            loss_meter_noisy.avg, psnr_meter_noisy.avg, ssim_meter_noisy.avg, lpips_meter_noisy.avg,
            noise_est_meter.avg
        )
        logger(msg, prefix='[valid]')


if __name__ == '__main__':
    # load args & configs
    parser = ArgumentParser(description='Blur Decomposition')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--config', default='./configs/cfg.yaml', help='path of config')
    parser.add_argument('--log_dir', default='log', help='path of log')
    parser.add_argument('--verbose', action='store_true', help='whether to print out logs')

    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.full_load(f)

    # Import blur decomposition dataset
    from data.dataset import BAistPP as BDDataset
    # DDP init
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    init_seeds(seed=rank)

    # Logger init
    if rank == 0:
        logger = Logger(file_path=join(args.log_dir, 'log_{}.txt'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))),
                        verbose=args.verbose)

    # Training model
    train(local_rank=args.local_rank,
          configs=configs,
          log_dir=args.log_dir)

    # Tear down the process group
    dist.destroy_process_group()