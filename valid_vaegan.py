import yaml
import random
import torch
import torchmetrics
import lpips
import time
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model.MBD import MBD, NoiseEstimationBranch
from model.utils import AverageMeter
from os.path import join
from logger import Logger
from model.bicyclegan.global_model import GuidePredictor as GP
from tqdm import tqdm
from torch.distributed.elastic.multiprocessing.errors import record
from data.dataset import BAistPP as BDDataset

loss_fn_alex = lpips.LPIPS(net='alex').to('cuda:0')
import torchvision.utils as vutils
# from NAFNet.basicsr.models import create_model

from restormer_arch import Restormer

import os

SAVE_DIR = "reconstruction_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation(local_rank, d_configs, p_configs, num_sampling, logger):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", local_rank)

    d_model = MBD(local_rank=local_rank, configs=d_configs)
    p_model = GP(local_rank=local_rank, configs=p_configs)
    #
    # noise_estimator = NoiseEstimationBranch()
    # # Load checkpoints
    # noise_estimator.load(d_configs['resume_dir'], device)
    # noise_estimator = noise_estimator.to(device).eval()



    if d_model.use_noise_aware:
        # Access the DDP-wrapped estimator
        noise_estimator = d_model.noise_estimator.module  # .module to unwrap DDP
    else:
        noise_estimator = None





    # noise_estimator = noise_estimator.cuda(local_rank).eval()

    # # In your eval code, after loading the noise estimator:
    # print("\n" + "=" * 60)
    # print("NOISE ESTIMATOR DEBUG INFO")
    # print("=" * 60)
    #
    # # Check if weights are initialized or trained
    # fc_weights = noise_estimator.fc[0].weight.data
    # fc_bias = noise_estimator.fc[0].bias.data
    #
    # print(f"FC layer 1 weight stats:")
    # print(f"  Mean: {fc_weights.mean().item():.6f}")
    # print(f"  Std: {fc_weights.std().item():.6f}")
    # print(f"  Min: {fc_weights.min().item():.6f}")
    # print(f"  Max: {fc_weights.max().item():.6f}")
    #
    # print(f"\nFC layer 1 bias: {fc_bias}")
    #
    # # Check last layer (the one that outputs noise level)
    # last_fc_weight = noise_estimator.fc[2].weight.data
    # last_fc_bias = noise_estimator.fc[2].bias.data
    #
    # print(f"\nLast FC layer weight: {last_fc_weight}")
    # print(f"Last FC layer bias: {last_fc_bias}")
    #
    # print("\nConv1 weight stats:")
    # conv1_weights = noise_estimator.conv1.weight.data
    # print(f"  Mean: {conv1_weights.mean().item():.6f}")
    # print(f"  Std: {conv1_weights.std().item():.6f}")
    #
    # print("=" * 60 + "\n")






    print("\n" + "=" * 60)
    print("Loading Restormer Denoiser...")
    print("=" * 60)

    try:
        denoiser = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False
        )

        checkpoint_path = 'pretrained_models/gaussian_color_denoising_sigma25.pth'
        checkpoint = torch.load(checkpoint_path,
                                map_location=f'cuda:{local_rank}')
        denoiser.load_state_dict(checkpoint['params'], strict=False)
        denoiser = denoiser.cuda(local_rank).eval()

        print(f"✅ Restormer loaded successfully!")
        print(f"   Model path: {checkpoint_path}")
        print(f"   Device: cuda:{local_rank}")

    except FileNotFoundError:
        print(
            "❌ Restormer weights not found at pretrained_models/gaussian_color_denoising_sigma25.pth")
        print(
            "   Download: wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_sigma25.pth -P pretrained_models/")
        exit(1)
    except Exception as e:
        print(f"❌ Failed to load Restormer: {e}")
        exit(1)

    print("=" * 60 + "\n")


    # dataset init
    dataset_args = d_configs['dataset_args']
    # noises = [5, 10, 20, 30, 40, 50]
    noises = [25]

    for noise_level in noises:
        print(f"[INFO] Testing with noise_level={noise_level}")

        dataset_args_override = dataset_args.copy()
        dataset_args_override['noisy'] = (noise_level > 0)
        dataset_args_override['noise_level'] = noise_level

        valid_dataset = BDDataset(set_type='valid', **dataset_args_override)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=1,
                                  num_workers=d_configs['num_workers'],
                                  pin_memory=True)
    evaluate(d_model, p_model, valid_loader, device, num_sampling, logger, valid_dataset.sigma, denoiser, noise_estimator)
    # evaluate(d_model, p_model, valid_loader, local_rank, num_sampling, logger, valid_dataset.sigma)

    print("\n" + "=" * 60)
    print("All evaluations complete!")
    print("=" * 60)
    logger.close()


@torch.no_grad()
def evaluate(d_model, p_model, valid_loader, device, num_sampling, logger, sigma, denoiser, noise_estimator):
    torch.cuda.empty_cache()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()
    psnr_meter_better = AverageMeter()
    ssim_meter_better = AverageMeter()
    lpips_meter_better = AverageMeter()

    # NEW: Add meters for tracking estimated noise
    noise_est_meter = AverageMeter()
    noise_error_meter = AverageMeter()

    time_stamp = time.time()

    NUM_SEQUENCES_TO_SAVE = 5

    noise_dir = join(SAVE_DIR, f"noise_sigma_{sigma}")
    os.makedirs(noise_dir, exist_ok=True)

    for i, tensor in enumerate(tqdm(valid_loader, total=len(valid_loader))):
        tensor['trend'] = torch.zeros_like(tensor['inp'])[:, :, :2]
        tensor['inp'] = tensor['inp'].to(device)  # (b, 1, 3, h, w)

        # LOG THE DATA RANGE
        print(f"\n=== Sample {i} ===")
        print(
            f"Input range: [{tensor['inp'].min():.2f}, {tensor['inp'].max():.2f}]")
        print(
            f"Input mean: {tensor['inp'].mean():.2f}, std: {tensor['inp'].std():.2f}")

        torch.cuda.empty_cache()

        blurry_input = tensor['inp'].squeeze(1)
        # Check data range and normalize if needed
        if blurry_input.max() > 1.0:
            # Data is in [0, 255] range
            print("Normalizing to 0 - 1")
            blurry_input_normalized = blurry_input.to(device) / 255.0
            # blurry_input_normalized = blurry_input_normalized.clamp(0.0, 1.0)
            needs_denorm = True
        else:
            # Data is already in [0, 1] range
            blurry_input_normalized = blurry_input.to(device)
            needs_denorm = False

        print("Input range to noise estimator:", blurry_input_normalized.min(),
              blurry_input_normalized.max())

        estimated_noise = noise_estimator(blurry_input_normalized).item()
        print("Estimated noise:", estimated_noise)

        # # === CORRECTED DEBUG ===
        # print("\n" + "=" * 60)
        # print("DETAILED FORWARD PASS DEBUG")
        # print("=" * 60)
        #
        # with torch.no_grad():
        #     x = blurry_input_normalized
        #     print(f"Input shape: {x.shape}")
        #
        #     # Conv layers
        #     x = noise_estimator.conv1(x)
        #     print(
        #         f"\nAfter conv1 (before ReLU): mean={x.mean():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
        #     x = F.relu(x)
        #     print(f"After conv1 ReLU: mean={x.mean():.4f}")
        #
        #     x = noise_estimator.conv2(x)
        #     print(
        #         f"\nAfter conv2 (before ReLU): mean={x.mean():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
        #     x = F.relu(x)
        #     print(f"After conv2 ReLU: mean={x.mean():.4f}")
        #
        #     # Pool
        #     x = noise_estimator.pool(x)
        #     x_flat = x.flatten(1)
        #     print(f"\nAfter pooling and flatten: shape={x_flat.shape}")
        #     print(f"  Pooled values: {x_flat[0]}")
        #
        #     # FC network step by step
        #     print(f"\n--- FC Network ---")
        #     x_fc1 = noise_estimator.fc[0](x_flat)  # Linear(16, 8)
        #     print(f"After FC1 Linear(16→8):")
        #     print(f"  Values: {x_fc1[0]}")
        #
        #     x_fc1_relu = noise_estimator.fc[1](x_fc1)  # ReLU
        #     print(f"After FC1 ReLU:")
        #     print(f"  Values: {x_fc1_relu[0]}")
        #     print(f"  Non-zero: {(x_fc1_relu > 0).sum().item()}/8")
        #
        #     x_fc2 = noise_estimator.fc[2](x_fc1_relu)  # Linear(8, 1) ← FIXED
        #     print(f"\nAfter FC2 Linear(8→1):")
        #     print(f"  Value: {x_fc2.item():.6f}")
        #
        #     x_sig = noise_estimator.fc[3](x_fc2)  # Sigmoid
        #     print(f"After Sigmoid:")
        #     print(f"  Value: {x_sig.item():.6f}")
        #
        #     x_final = x_sig * 25.0
        #     print(f"Final (* 25):")
        #     print(f"  Value: {x_final.item():.6f}")
        #
        # print("=" * 60 + "\n")

        # # Compare with actual forward
        # estimated_noise = noise_estimator(blurry_input_normalized).item()
        # print(f"Actual forward() result: {estimated_noise:.6f}\n")


        noise_threshold = 20
        # needs_denoiser = False

        if estimated_noise > noise_threshold:
            print("Noise level is too high!, estimated: ", estimated_noise)

            # needs_denoiser = True

            # Apply Restormer denoising
            denoised_blur = denoiser(blurry_input_normalized)

            # Denormalize back to original range if needed
            if needs_denorm:
                denoised_blur = denoised_blur * 255.0

            # Restore the expected shape: (b, 3, h, w) -> (b, 1, 3, h, w)
            tensor['inp'] = denoised_blur.unsqueeze(1)



        # if i == 0:
        #     print(
        #         f"\n[Denoising] Input range: [{blurry_input.min():.2f}, {blurry_input.max():.2f}]")
        #     print(
        #         f"[Denoising] Output range: [{denoised_blur.min():.2f}, {denoised_blur.max():.2f}]")
        #     print(f"[Denoising] Normalized for denoiser: {needs_denorm}\n")



        tensor['gt'] = tensor['gt'].to(device)
        b, num_gts, c, h, w = tensor['gt'].shape

        # # NEW: Actually ADD noise to the input images if sigma > 0
        # if sigma > 0:
        #     # Get the blurry input (b, 1, 3, h, w)
        #     noisy_inp = tensor['inp'].clone()
        #     # Normalize to [0, 1]
        #     noisy_inp = noisy_inp / 255.0
        #     # Add Gaussian noise with the specified sigma
        #     noise = torch.randn_like(noisy_inp) * (sigma / 255.0)
        #     noisy_inp = noisy_inp + noise
        #     # Clamp and scale back to [0, 255]
        #     noisy_inp = torch.clamp(noisy_inp, 0, 1) * 255.0
        #     tensor['inp'] = noisy_inp
        #
        # # Set the ground truth noise level
        # tensor['noise_level'] = torch.full((b, 1), float(sigma), device=device)
        #
        # if i < NUM_SEQUENCES_TO_SAVE:
        #     seq_dir = join(noise_dir, f"sequence_{i}")
        #     os.makedirs(seq_dir, exist_ok=True)
        #
        #     input_img = tensor['inp'][0, 0].cpu()
        #     input_save_path = join(seq_dir, "00_input_blurry.png")
        #     vutils.save_image(input_img / 255.0, input_save_path)

        psnr_vals = []
        ssim_vals = []
        lpips_vals = []
        psnr_vals_better = []
        ssim_vals_better = []
        lpips_vals_better = []

        for _ in range(num_sampling):
            p_tensor = {}
            p_tensor['inp'] = F.interpolate(tensor['inp'], size=(3, 256, 256))
            p_tensor['trend'] = F.interpolate(tensor['trend'], size=(2, 256, 256))
            p_tensor['video'] = ''
            out_tensor = p_model.test(inp_tensor=p_tensor, zeros=False)
            tensor['trend'] = out_tensor['pred_trends'].to(device)
            tensor['trend'] = F.interpolate(tensor['trend'], size=(h, w))[None]
            out_tensor = d_model.update(inp_tensor=tensor, training=False)
            pred_imgs = out_tensor['pred_imgs']
            gt_imgs = out_tensor['gt_imgs']

            # NEW: Extract estimated noise from output tensor
            if 'noise_level_est' in out_tensor:
                est_noise_batch = out_tensor['noise_level_est']  # (b, 1)
                est_noise_avg = est_noise_batch.mean().item()
                noise_est_meter.update(est_noise_avg, b)

                # Calculate error compared to ground truth sigma
                noise_error = abs(est_noise_avg - sigma)
                noise_error_meter.update(noise_error, b)

                if i == 0 and _ == 0:
                    print(f"[DEBUG] First batch - GT noise: {sigma}, Estimated: {est_noise_avg:.2f}")
            else:
                print(f"[WARNING] 'estimated_noise' not found in out_tensor! Keys: {out_tensor.keys()}")

            pred_imgs_for_save = pred_imgs[0]
            gt_imgs_for_save = gt_imgs[0]

            if i < NUM_SEQUENCES_TO_SAVE:
                seq_dir = join(noise_dir, f"sequence_{i}")
                os.makedirs(seq_dir, exist_ok=True)
                for frame_idx in range(num_gts):
                    pred_frame = pred_imgs_for_save[frame_idx].cpu() / 255.0
                    pred_path = join(seq_dir, f"frame{frame_idx}_predicted.png")
                    vutils.save_image(pred_frame, pred_path)

                    gt_frame = gt_imgs_for_save[frame_idx].cpu() / 255.0
                    gt_path = join(seq_dir, f"frame{frame_idx}_groundtruth.png")
                    vutils.save_image(gt_frame, gt_path)

            pred_imgs = pred_imgs.reshape(num_gts * b, c, h, w)
            pred_imgs = pred_imgs.to(device)
            gt_imgs = gt_imgs.to(device)
            pred_imgs_rev = torch.flip(pred_imgs, dims=[1, ])
            pred_imgs = pred_imgs.reshape(num_gts * b, c, h, w)
            pred_imgs_rev = pred_imgs_rev.reshape(num_gts * b, c, h, w)
            gt_imgs = gt_imgs.reshape(num_gts * b, c, h, w)
            psnr_val = torchmetrics.functional.psnr(pred_imgs, gt_imgs, data_range=255)
            psnr_val_rev = torchmetrics.functional.psnr(pred_imgs_rev, gt_imgs, data_range=255)
            ssim_val = torchmetrics.functional.ssim(pred_imgs, gt_imgs, data_range=255)
            ssim_val_rev = torchmetrics.functional.ssim(pred_imgs_rev, gt_imgs, data_range=255)
            pred_imgs = (pred_imgs - (255. / 2)) / (255. / 2)
            pred_imgs_rev = (pred_imgs_rev - (255. / 2)) / (255. / 2)
            gt_imgs = (gt_imgs - (255. / 2)) / (255. / 2)
            lpips_val = loss_fn_alex(pred_imgs, gt_imgs)
            lpips_val_rev = loss_fn_alex(pred_imgs_rev, gt_imgs)

            psnr_vals.append(psnr_val)
            psnr_vals_better.append(max(psnr_val, psnr_val_rev))
            ssim_vals.append(ssim_val)
            ssim_vals_better.append(max(ssim_val, ssim_val_rev))
            lpips_vals.append(lpips_val.mean().detach())
            lpips_vals_better.append(min(lpips_val.mean().detach(), lpips_val_rev.mean().detach()))

        psnr_meter.update(max(psnr_vals), num_gts * b)
        ssim_meter.update(max(ssim_vals), num_gts * b)
        lpips_meter.update(min(lpips_vals), num_gts * b)
        psnr_meter_better.update(max(psnr_vals_better), num_gts * b)
        ssim_meter_better.update(max(ssim_vals_better), num_gts * b)
        lpips_meter_better.update(min(lpips_vals_better), num_gts * b)

    eval_time_interval = time.time() - time_stamp

    # NEW: Print results with estimated noise information
    msg = 'eval time: {} sec, psnr: {:.5f}, ssim: {:.5f}, lpips: {:.4f}, est_noise: {:.2f} (error: {:.2f})'.format(
        eval_time_interval, psnr_meter.avg, ssim_meter.avg, lpips_meter.avg,
        noise_est_meter.avg, noise_error_meter.avg
    )
    logger(msg, prefix=f'[valid σ={sigma}]')

    msg = 'eval time: {:.4f} sec, psnr: {:.4f}, ssim: {:.4f}, lpips: {:.4f}, est_noise: {:.2f} (error: {:.2f})'.format(
        eval_time_interval, psnr_meter_better.avg, ssim_meter_better.avg, lpips_meter_better.avg,
        noise_est_meter.avg, noise_error_meter.avg
    )
    logger(msg, prefix=f'[valid max. σ={sigma}]')

    print(f"[INFO] Saved images for σ={sigma} in {noise_dir}")
    print(
        f"[INFO] Noise Estimation - GT: {sigma:.1f}, Estimated: {noise_est_meter.avg:.2f}, Error: {noise_error_meter.avg:.2f}")

@record
def main():
    parser = ArgumentParser(description='Guidance prediction & Blur Decomposition')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--log_dir', default='logs', help='path of log')
    parser.add_argument('--log_name', default='valid', help='log name')
    parser.add_argument('--predictor_resume_dir', help='path of log', required=True)
    parser.add_argument('--decomposer_resume_dir', help='path of log', required=True)
    parser.add_argument('--data_dir', nargs='+', required=True)
    parser.add_argument('--num_iters', type=int, default=1, help='number of iters')
    parser.add_argument('-ns', '--num_sampling', type=int, default=1, help='number of sampling times')
    parser.add_argument('--verbose', action='store_true', help='whether to print out logs')
    args = parser.parse_args()
    num_sampling = args.num_sampling

    decomposer_config = join(args.decomposer_resume_dir, 'cfg.yaml')
    with open(decomposer_config) as f:
        d_configs = yaml.full_load(f)
    d_configs['resume_dir'] = args.decomposer_resume_dir
    d_configs['num_iterations'] = args.num_iters

    predictor_config = join(args.predictor_resume_dir, 'cfg.yaml')
    with open(predictor_config, 'rt', encoding='utf8') as f:
        p_configs = yaml.full_load(f)
    p_configs['bicyclegan_args']['checkpoints_dir'] = args.predictor_resume_dir
    p_configs['bicyclegan_args']['isTrain'] = False
    p_configs['bicyclegan_args']['epoch'] = 'latest'
    p_configs['bicyclegan_args']['no_encode'] = False

    is_gen_blur = True
    for root_dir in d_configs['dataset_args']['root_dir']:
        if 'b-aist++' in root_dir:
            is_gen_blur = False

        d_configs['dataset_args']['aug_args']['valid']['image'] = {
            'NearBBoxResizedSafeCrop': {'max_ratio': 0, 'height': 192, 'width': 160}
        }
    d_configs['dataset_args']['root_dir'] = args.data_dir
    d_configs['dataset_args']['use_trend'] = False

    logger = Logger(file_path=join(args.log_dir, '{}.txt'.format(args.log_name)), verbose=args.verbose)
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    init_seeds(seed=rank)

    validation(local_rank=args.local_rank, d_configs=d_configs, p_configs=p_configs, num_sampling=num_sampling,
               logger=logger)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
