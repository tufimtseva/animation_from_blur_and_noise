import yaml
import torch
import torch.distributed as dist
import torchmetrics
import lpips
import time
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model.MBD import MBD
from model.utils import AverageMeter
from os.path import join
from logger import Logger
from data.dataset import BAistPP as BDDataset
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, valid_loader, local_rank):
    device = torch.device("cuda", local_rank)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()
    
    for i, tensor in enumerate(tqdm(valid_loader)):
        tensor['inp'] = tensor['inp'].to(device)
        tensor['gt'] = tensor['gt'].to(device)
        b, num_gts, c, h, w = tensor['gt'].shape
        
        out_tensor = model.update(inp_tensor=tensor, training=False)
        pred_imgs = out_tensor['pred_imgs'].reshape(num_gts * b, c, h, w)
        gt_imgs = out_tensor['gt_imgs'].reshape(num_gts * b, c, h, w)
        
        psnr_val = torchmetrics.functional.psnr(pred_imgs, gt_imgs, data_range=255)
        ssim_val = torchmetrics.functional.ssim(pred_imgs, gt_imgs, data_range=255)
        
        pred_imgs_norm = (pred_imgs - 127.5) / 127.5
        gt_imgs_norm = (gt_imgs - 127.5) / 127.5
        lpips_val = loss_fn_alex(pred_imgs_norm, gt_imgs_norm).mean()
        
        psnr_meter.update(psnr_val.item(), num_gts * b)
        ssim_meter.update(ssim_val.item(), num_gts * b)
        lpips_meter.update(lpips_val.item(), num_gts * b)
    
    print(f'PSNR: {psnr_meter.avg:.4f}, SSIM: {ssim_meter.avg:.4f}, LPIPS: {lpips_meter.avg:.4f}')

def main():
    parser = ArgumentParser()
    parser.add_argument('--decomposer_resume_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()
    
    # Initialize DDP
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    
    config_path = join(args.decomposer_resume_dir, 'cfg.yaml')
    with open(config_path) as f:
        configs = yaml.full_load(f)
    
    configs['resume_dir'] = args.decomposer_resume_dir
    configs['dataset_args']['root_dir'] = [args.data_dir]
    configs['dataset_args']['use_trend'] = False
    
    model = MBD(local_rank=args.local_rank, configs=configs)
    
    valid_dataset = BDDataset(set_type='valid', **configs['dataset_args'])
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4, pin_memory=True)
    
    evaluate(model, valid_loader, args.local_rank)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
