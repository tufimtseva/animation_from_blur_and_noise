"""
Changes from original:
1. Added temporal consistency loss 
2. Perceptual loss already exists ( enable via perc_ratio in config)
"""

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from os.path import join
from model.flow_estimator import FlowEstimator
from model.decomposer import Decomposer
from model.embedder import get_embedder
from model.utils import ckpt_convert, Vgg19
from torch.nn.parallel import DistributedDataParallel as DDP


class TemporalConsistencyLoss(nn.Module):
    """
    Enforces smooth motion between consecutive frames.
    Penalizes sudden jumps/jitter in the predicted sequence.
    """
    def __init__(self, order=1):
        super().__init__()
        self.order = order  # 1=velocity smoothness, 2=acceleration smoothness
    
    def forward(self, pred_sequence):
        """
        Args:
            pred_sequence: (B, T, C, H, W) - predicted frame sequence
        Returns:
            Temporal consistency loss value
        """
        B, T, C, H, W = pred_sequence.shape
        
        # First-order: penalize large frame-to-frame differences
        diff1 = pred_sequence[:, 1:] - pred_sequence[:, :-1]  # (B, T-1, C, H, W)
        loss = torch.mean(diff1 ** 2)
        
        # Second-order: penalize changes in velocity (acceleration)
        if self.order >= 2 and T > 2:
            diff2 = diff1[:, 1:] - diff1[:, :-1]  # (B, T-2, C, H, W)
            loss = loss + 0.5 * torch.mean(diff2 ** 2)
        
        return loss


class MBD:
    """
    Multimodal blur decomposition - MODIFIED VERSION
    """

    def __init__(self, local_rank, configs):
        self.configs = configs
        self.val_range = configs['val_range']
        self.num_iters = configs['num_iterations']
        self.num_gts = configs['dataset_args']['num_gts']
        self.hybrid = configs.get('hybrid', True)
        self.residual = configs.get('residual', False)
        self.residual_blur = configs.get('residual_blur', False)
        self.flow_to_s2 = configs.get('flow_to_s2', True)
        self.s1_to_s2 = configs.get('s1_to_s2', False)
        self.use_trend = configs.get('use_trend', True)
        
        # === LOSS WEIGHTS (existing + new) ===
        self.perc_ratio = configs.get('perc_ratio', 0)  # Existing: perceptual loss weight
        self.temporal_ratio = configs.get('temporal_ratio', 0)  # NEW: temporal consistency weight
        self.temporal_order = configs.get('temporal_order', 1)  # NEW: 1 or 2
        
        self.device = torch.device("cuda", local_rank)

        # Init modules
        if self.num_iters > 0:
            self.flow_estimator = FlowEstimator(device=self.device, **configs['flow_estimator_args'])
        self.trend_embedder, trend_embed_dim = get_embedder(**configs['trend_embedder_args'])
        if self.use_trend:
            configs['decomposer_s1_args']['in_channels'] += trend_embed_dim
        self.flow_embbedder, flow_embed_dim = get_embedder(**configs['flow_embedder_args'])
        if self.flow_to_s2:
            configs['decomposer_s2_args']['in_channels'] += (self.num_gts - 1) * flow_embed_dim
        if self.s1_to_s2:
            configs['decomposer_s2_args']['in_channels'] += self.num_gts * 3
        self.decomposer_s1 = Decomposer(**configs['decomposer_s1_args'])
        self.decomposer_s2 = Decomposer(**configs['decomposer_s2_args'])

        # Replace BN as SyncBN
        self.decomposer_s1 = nn.SyncBatchNorm.convert_sync_batchnorm(self.decomposer_s1)
        self.decomposer_s2 = nn.SyncBatchNorm.convert_sync_batchnorm(self.decomposer_s2)

        # Move modules to GPU
        self.decomposer_s1.to(device=self.device)
        self.decomposer_s2.to(device=self.device)

        # Load checkpoints
        if (local_rank == 0) and (configs['resume_dir'] is not None):
            self.load_model(configs['resume_dir'], self.device)

        # DDP wrapper
        self.decomposer_s1 = DDP(self.decomposer_s1, device_ids=[local_rank], output_device=local_rank,
                                 broadcast_buffers=False)
        self.decomposer_s2 = DDP(self.decomposer_s2, device_ids=[local_rank], output_device=local_rank,
                                 broadcast_buffers=False)

        # Init optimizer, learning rate scheduler, and loss function
        self.optimizer = optim.AdamW(itertools.chain(self.decomposer_s1.parameters(),
                                                     self.decomposer_s2.parameters()), **configs['optimizer'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs['epoch'],
                                                              **configs['scheduler'])
        self.l2 = nn.MSELoss()
        
        # Perceptual loss (existing)
        if self.perc_ratio > 0:
            self.vgg = Vgg19().to(self.device)
        
        # === NEW: Temporal consistency loss ===
        if self.temporal_ratio > 0:
            self.temporal_loss = TemporalConsistencyLoss(order=self.temporal_order)

    def train(self):
        self.decomposer_s1.train()
        self.decomposer_s2.train()

    def eval(self):
        self.decomposer_s1.eval()
        self.decomposer_s2.eval()

    def scheduler_step(self):
        self.scheduler.step()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def load_model(self, log_dir, device, name=None):
        if name is None:
            self.decomposer_s1.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s1.pth'), map_location=device)))
            self.decomposer_s2.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s2.pth'), map_location=device)))
        else:
            self.decomposer_s1.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s1_{}.pth'.format(name)), map_location=device)))
            self.decomposer_s2.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s2_{}.pth'.format(name)), map_location=device)))

    def save_model(self, log_dir, name=None):
        if name is None:
            torch.save(self.decomposer_s1.state_dict(), join(log_dir, 'decomposer_s1.pth'))
            torch.save(self.decomposer_s2.state_dict(), join(log_dir, 'decomposer_s2.pth'))
        else:
            torch.save(self.decomposer_s1.state_dict(), join(log_dir, 'decomposer_s1_{}.pth'.format(name)))
            torch.save(self.decomposer_s2.state_dict(), join(log_dir, 'decomposer_s2_{}.pth'.format(name)))

    def update(self, inp_tensor, hybrid_flag=0, training=True):
        """
        Forward and backward propagation - MODIFIED with temporal loss
        """
        if training:
            self.train()
        else:
            self.eval()
        
        out_tensor = {}
        blur_img = inp_tensor['inp']
        sharp_imgs = inp_tensor['gt']
        trend_img = inp_tensor.get('trend', None)
        b, n, c, h, w = blur_img.shape
        blur_img = blur_img.reshape(b, n * c, h, w)
        
        if trend_img is not None:
            b_t, n_t, tc, h_t, w_t = trend_img.shape
            trend_img = trend_img.reshape(b_t, n_t * tc, h_t, w_t)
        blur_img = blur_img / self.val_range
        sharp_imgs = sharp_imgs / self.val_range

        # Forward propagation
        all_pred_imgs = []
        
        # First stage
        if self.residual_blur:
            pred_imgs = self.first_stage(blur_img, trend_img) + blur_img.unsqueeze(dim=1)
        else:
            pred_imgs = self.first_stage(blur_img, trend_img)
        all_pred_imgs.append(pred_imgs)

        # Refinement iterations
        for _ in range(self.num_iters):
            if training:
                if (hybrid_flag == 0) or (not self.hybrid):
                    if self.residual:
                        pred_imgs = pred_imgs + self.second_stage(blur_img, pred_imgs)
                    else:
                        pred_imgs = self.second_stage(blur_img, pred_imgs)
                elif hybrid_flag == 1:
                    if self.residual:
                        pred_imgs = pred_imgs + self.second_stage(blur_img, sharp_imgs)
                    else:
                        pred_imgs = self.second_stage(blur_img, sharp_imgs)
                else:
                    raise ValueError
            else:
                if self.residual:
                    pred_imgs = pred_imgs + self.second_stage(blur_img, pred_imgs)
                else:
                    pred_imgs = self.second_stage(blur_img, pred_imgs)
            all_pred_imgs.append(pred_imgs)

        # ============================================================
        # CALCULATE LOSSES - MODIFIED SECTION
        # ============================================================
        loss = 0.
        loss_l2 = 0.
        loss_perc = 0.
        loss_temporal = 0.
        
        for pred_imgs in all_pred_imgs:
            b, num_gts, c, h, w = pred_imgs.shape
            
            # 1. Reconstruction loss (L2) - ORIGINAL
            l2_loss = self.l2(pred_imgs, sharp_imgs)
            loss += l2_loss
            loss_l2 += l2_loss.item()
            
            # 2. Perceptual loss - EXISTING (enable via perc_ratio > 0)
            if self.perc_ratio > 0:
                x_vgg = self.vgg(pred_imgs.reshape(b * num_gts, c, h, w))
                y_vgg = self.vgg(sharp_imgs.reshape(b * num_gts, c, h, w))
                perc_loss = 0.
                for i in range(len(x_vgg)):
                    perc_loss += torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                loss += self.perc_ratio * perc_loss
                loss_perc += perc_loss.item()
            
            # 3. Temporal consistency loss - NEW
            if self.temporal_ratio > 0:
                temp_loss = self.temporal_loss(pred_imgs)
                loss += self.temporal_ratio * temp_loss
                loss_temporal += temp_loss.item()

        # Backward propagation
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Register tensors
        out_tensor['loss'] = loss
        out_tensor['loss_l2'] = loss_l2  # NEW: for logging
        out_tensor['loss_perc'] = loss_perc  # NEW: for logging
        out_tensor['loss_temporal'] = loss_temporal  # NEW: for logging
        out_tensor['inp_img'] = self.val_range * blur_img[:, :c].detach()
        if trend_img is not None:
            out_tensor['trend_img'] = trend_img.detach()
        out_tensor['pred_imgs'] = torch.clamp(self.val_range * all_pred_imgs[-1].detach(), min=0, max=self.val_range)
        out_tensor['gt_imgs'] = self.val_range * sharp_imgs

        return out_tensor

    def first_stage(self, blur_img, trend_img):
        """First stage: coarse decomposition"""
        if not self.use_trend:
            return self.decomposer_s1(blur_img)

        num_gts = self.configs['dataset_args']['num_gts']
        b, tc, h, w = trend_img.shape
        trend_static = torch.zeros(b, 1, h, w).to(self.device)
        trend_static[(trend_img[:, 0:1] == 0) & (trend_img[:, 1:2] == 0)] = 1
        trend_static = trend_static.unsqueeze(dim=1)
        trend_dynamic = 1 - trend_static

        inp_img = torch.cat([blur_img, trend_img], dim=1)
        pred_imgs = trend_dynamic * self.decomposer_s1(inp_img)
        pred_imgs += trend_static * blur_img.unsqueeze(dim=1).repeat(1, num_gts, 1, 1, 1)
        torch.cuda.synchronize()
        return pred_imgs

    def second_stage(self, blur_img, pred_imgs):
        """Second stage: refinement"""
        b, num_gts, c, h, w = pred_imgs.shape
        tensor_inp = [blur_img, ]

        if self.s1_to_s2:
            tensor_inp.append(pred_imgs.reshape(b, num_gts * c, h, w))

        if self.flow_to_s2:
            ref_imgs = pred_imgs[:, :-1]
            ref_imgs = self.val_range * ref_imgs.reshape(b * (num_gts - 1), c, h, w)
            tgt_imgs = pred_imgs[:, 1:]
            tgt_imgs = self.val_range * tgt_imgs.reshape(b * (num_gts - 1), c, h, w)
            flows = self.flow_estimator.batch_multi_inference(ref_imgs=ref_imgs, tgt_imgs=tgt_imgs)
            flows = flows.reshape(b, (num_gts - 1), 2, h, w)
            flows = flows.reshape(b, (num_gts - 1) * 2, h, w)
            tensor_inp.append(flows)

        tensor_inp = torch.cat(tensor_inp, dim=1)
        tensor_out = self.decomposer_s2(tensor_inp)
        return tensor_out

    @torch.no_grad()
    def inference(self, blur_img, trend, num_iters=0, full_results=False):
        """Inference method - unchanged"""
        self.eval()
        h, w, _ = blur_img.shape
        blur_img = torch.from_numpy(blur_img).float().to(self.device)
        blur_img = blur_img.permute(2, 0, 1).unsqueeze(dim=0) / self.val_range

        if len(trend.shape) == 1:
            trend = torch.from_numpy(trend).float().to(self.device)
            trend = trend[None, None, None].repeat(1, h, w, 1).permute(0, 3, 1, 2)
        elif len(trend.shape) == 3:
            trend = torch.from_numpy(trend).float().to(self.device)
            trend = trend[None].permute(0, 3, 1, 2)
        else:
            raise NotImplementedError

        pred_imgs_s1 = self.first_stage(blur_img=blur_img, trend_img=trend)
        torch.cuda.synchronize()
        for i in range(num_iters):
            if self.residual:
                pred_residual = self.second_stage(blur_img=blur_img, pred_imgs=pred_imgs_s1)
                pred_imgs_s2 = pred_imgs_s1 + pred_residual
            else:
                pred_imgs_s2 = self.second_stage(blur_img=blur_img, pred_imgs=pred_imgs_s1)
            pred_imgs_s1 = pred_imgs_s2
        if num_iters == 0:
            pred_imgs_s2 = pred_imgs_s1
        pred_imgs_s2 = pred_imgs_s2.squeeze(dim=0)
        pred_imgs_s2 = torch.clamp(pred_imgs_s2 * self.val_range, 0, 255)
        pred_imgs_s2 = pred_imgs_s2.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

        if not full_results:
            return pred_imgs_s2
        else:
            pred_imgs_s1 = pred_imgs_s1.squeeze(dim=0)
            pred_imgs_s1 = torch.clamp(pred_imgs_s1 * self.val_range, 0, 255)
            pred_imgs_s1 = pred_imgs_s1.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
            return pred_imgs_s2, pred_imgs_s1
