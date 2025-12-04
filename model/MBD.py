import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from os.path import join
from model.flow_estimator import FlowEstimator
from model.decomposer import Decomposer
from model.embedder import get_embedder
from model.utils import ckpt_convert, Vgg19
from torch.nn.parallel import DistributedDataParallel as DDP

class NoiseEstimationBranch(nn.Module):
    """Tiny network to estimate noise level from blurry input image!"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input blurry image (B, 3, H, W) - already normalized to [0, 1]
        Returns:
            noise_level: Estimated noise level (B, 1) in range [0, 25]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        noise_level = self.fc(x) * 25.0  # Scale to [0, 25]
        return noise_level


class NoiseModulatedDecomposer(nn.Module):
    """
    Wraps a Decomposer and adds FiLM-style noise-aware feature modulation
    """

    def __init__(self, base_decomposer, feature_channels):
        super().__init__()
        self.decomposer = base_decomposer
        self.feature_channels = feature_channels

        # FiLM modulation layers
        self.gamma_layer = nn.Linear(1, feature_channels)  # Scale
        self.beta_layer = nn.Linear(1, feature_channels)  # Shift

        # Store noise level for this forward pass
        self.current_noise_level = None

        # Register forward hook on first_conv to apply FiLM modulation
        self.decomposer.first_conv.register_forward_hook(self._film_hook)

    def set_noise_level(self, noise_level):
        """Store noise level for the next forward pass"""
        self.current_noise_level = noise_level

    def _film_hook(self, module, input, output):
        """
        Forward hook that applies FiLM modulation to features
        This is called automatically after first_conv
        """
        if self.current_noise_level is None:
            return output  # No modulation

        B = self.current_noise_level.size(0)
        # Compute gamma and beta from noise level
        gamma = self.gamma_layer(self.current_noise_level).view(B, self.feature_channels, 1, 1)
        beta = self.beta_layer(self.current_noise_level).view(B, self.feature_channels, 1, 1)

        # Apply FiLM: modulated_features = gamma * features + beta
        modulated_output = gamma * output + beta

        return modulated_output

    def forward(self, x):
        """Forward pass through the decomposer with FiLM modulation applied via hook"""
        return self.decomposer(x)


class MBD:
    """
    Multimodal blur decomposition with Noise-Aware Feature Modulation (FiLM)
    """

    def __init__(self, local_rank, configs):
        self.configs = configs
        self.val_range = configs['val_range']
        self.num_iters = configs['num_iterations']
        self.num_gts = configs['dataset_args']['num_gts']
        self.hybrid = True
        if 'hybrid' in configs:
            self.hybrid = configs['hybrid']
        self.residual = False
        if 'residual' in configs:
            self.residual = configs['residual']
        self.residual_blur = False
        if 'residual_blur' in configs:
            self.residual_blur = configs['residual_blur']
        self.flow_to_s2 = True
        if 'flow_to_s2' in configs:
            self.flow_to_s2 = configs['flow_to_s2']
        self.s1_to_s2 = False
        if 's1_to_s2' in configs:
            self.s1_to_s2 = configs['s1_to_s2']
        self.use_trend = True
        if 'use_trend' in configs:
            self.use_trend = configs['use_trend']
        self.perc_ratio = 0
        if 'perc_ratio' in configs:
            self.perc_ratio = configs['perc_ratio']

        # NEW: Noise-aware training flag
        self.use_noise_aware = True
        if 'use_noise_aware' in configs:
            self.use_noise_aware = configs['use_noise_aware']

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

        # Create base decomposers
        base_decomposer_s1 = Decomposer(**configs['decomposer_s1_args'])
        base_decomposer_s2 = Decomposer(**configs['decomposer_s2_args'])

        # NEW: Wrap decomposers with noise-aware FiLM modulation
        if self.use_noise_aware:
            self.noise_estimator = NoiseEstimationBranch()
            self.noise_estimator.to(device=self.device)

            # Get feature channels from config
            feature_channels = configs['decomposer_s1_args']['block_expansion']

            # Wrap both decomposers with FiLM modulation
            self.decomposer_s1 = NoiseModulatedDecomposer(base_decomposer_s1, feature_channels)
            self.decomposer_s2 = NoiseModulatedDecomposer(base_decomposer_s2, feature_channels)

            print(f"[MBD] ✓ Noise-aware FiLM modulation ENABLED (feature_channels={feature_channels})")
        else:
            self.decomposer_s1 = base_decomposer_s1
            self.decomposer_s2 = base_decomposer_s2
            self.noise_estimator = None
            print("[MBD] Noise-aware feature modulation DISABLED")

        # Replace BN as SyncBN
        self.decomposer_s1 = nn.SyncBatchNorm.convert_sync_batchnorm(self.decomposer_s1)
        self.decomposer_s2 = nn.SyncBatchNorm.convert_sync_batchnorm(self.decomposer_s2)
        if self.use_noise_aware:
            self.noise_estimator = nn.SyncBatchNorm.convert_sync_batchnorm(self.noise_estimator)

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
        if self.use_noise_aware:
            self.noise_estimator = DDP(self.noise_estimator, device_ids=[local_rank], output_device=local_rank,
                                       broadcast_buffers=False)

        # Init optimizer, learning rate scheduler, and loss function
        params_to_optimize = itertools.chain(self.decomposer_s1.parameters(),
                                             self.decomposer_s2.parameters())
        if self.use_noise_aware:
            params_to_optimize = itertools.chain(params_to_optimize, self.noise_estimator.parameters())

        self.optimizer = optim.AdamW(params_to_optimize, **configs['optimizer'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs['epoch'],
                                                              **configs['scheduler'])
        self.l2 = nn.MSELoss()
        if self.perc_ratio > 0:
            self.vgg = Vgg19().to(self.device)

        # NEW: Loss weight for noise estimation
        self.noise_loss_weight = 0.1
        if 'noise_loss_weight' in configs:
            self.noise_loss_weight = configs['noise_loss_weight']

    def train(self):
        """Open training mode"""
        self.decomposer_s1.train()
        self.decomposer_s2.train()
        if self.use_noise_aware:
            self.noise_estimator.train()

    def eval(self):
        """Open evaluating mode"""
        self.decomposer_s1.eval()
        self.decomposer_s2.eval()
        if self.use_noise_aware:
            self.noise_estimator.eval()

    def scheduler_step(self):
        """Update scheduler after each epoch"""
        self.scheduler.step()

    def get_lr(self):
        """Get learning rate"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def load_model(self, log_dir, device, name=None):
        """Load checkpoints for decomposer_s1, decomposer_s2, and noise_estimator"""
        if name is None:
            self.decomposer_s1.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s1.pth'), map_location=device)))
            self.decomposer_s2.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s2.pth'), map_location=device)))
            if self.use_noise_aware:
                try:
                    self.noise_estimator.load_state_dict(
                        ckpt_convert(torch.load(join(log_dir, 'noise_estimator.pth'), map_location=device)))
                    print("[MBD] ✓ Loaded noise estimator checkpoint")
                except:
                    print("[MBD] ⚠ No noise estimator checkpoint found, starting from scratch")
        else:
            self.decomposer_s1.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s1_{}.pth'.format(name)), map_location=device)))
            self.decomposer_s2.load_state_dict(
                ckpt_convert(torch.load(join(log_dir, 'decomposer_s2_{}.pth'.format(name)), map_location=device)))
            if self.use_noise_aware:
                try:
                    self.noise_estimator.load_state_dict(
                        ckpt_convert(
                            torch.load(join(log_dir, 'noise_estimator_{}.pth'.format(name)), map_location=device)))
                    print("[MBD] ✓ Loaded noise estimator checkpoint")
                except:
                    print("[MBD] ⚠ No noise estimator checkpoint found, starting from scratch")

    def save_model(self, log_dir, name=None):
        """Save checkpoints for decomposer_s1, decomposer_s2, and noise_estimator"""
        if name is None:
            torch.save(self.decomposer_s1.state_dict(), join(log_dir, 'decomposer_s1.pth'))
            torch.save(self.decomposer_s2.state_dict(), join(log_dir, 'decomposer_s2.pth'))
            if self.use_noise_aware:
                torch.save(self.noise_estimator.state_dict(), join(log_dir, 'noise_estimator.pth'))
        else:
            torch.save(self.decomposer_s1.state_dict(), join(log_dir, 'decomposer_s1_{}.pth'.format(name)))
            torch.save(self.decomposer_s2.state_dict(), join(log_dir, 'decomposer_s2_{}.pth'.format(name)))
            if self.use_noise_aware:
                torch.save(self.noise_estimator.state_dict(), join(log_dir, 'noise_estimator_{}.pth'.format(name)))

    def update(self, inp_tensor, hybrid_flag=0, training=True):
        """
        Forward propagation, and backward propagation (if training == True) for a batch of data
        """
        # Preparation
        if training:
            self.train()
        else:
            self.eval()
        out_tensor = {}
        blur_img, sharp_imgs, trend_img = inp_tensor['inp'], inp_tensor['gt'], inp_tensor['trend']

        # NEW: Get noise level (either ground truth or estimated)
        gt_noise_level = inp_tensor.get('noise_level', None)

        b, n, c, h, w = blur_img.shape
        blur_img = blur_img.reshape(b, n * c, h, w)  # shape (b, 1 * 3, h, w)
        b, n, tc, h, w = trend_img.shape
        trend_img = trend_img.reshape(b, n * tc, h, w)  # shape (b, 1 * 2, h, w)
        blur_img = blur_img / self.val_range
        sharp_imgs = sharp_imgs / self.val_range

        # NEW: Estimate noise level
        estimated_noise_level = None
        noise_estimation_loss = torch.tensor(0.0).to(self.device)

        if self.use_noise_aware:
            # Estimate noise from blurry input
            estimated_noise_level = self.noise_estimator(blur_img)  # (B, 1)

            # If ground truth noise is provided, calculate noise estimation loss
            if gt_noise_level is not None:
                noise_estimation_loss = F.l1_loss(estimated_noise_level, gt_noise_level)

            # Store for logging
            out_tensor['estimated_noise'] = estimated_noise_level.detach()
            if gt_noise_level is not None:
                out_tensor['gt_noise'] = gt_noise_level.detach()

        # Forward propagation
        all_pred_imgs = []

        # First stage
        if self.residual_blur:
            pred_imgs = self.first_stage(blur_img, trend_img, estimated_noise_level) + blur_img.unsqueeze(dim=1)
        else:
            pred_imgs = self.first_stage(blur_img, trend_img, estimated_noise_level)
        all_pred_imgs.append(pred_imgs)

        for _ in range(self.num_iters):
            if training:
                if (hybrid_flag == 0) or (not self.hybrid):
                    if self.residual:
                        pred_imgs = pred_imgs + self.second_stage(blur_img, pred_imgs, estimated_noise_level)
                    else:
                        pred_imgs = self.second_stage(blur_img, pred_imgs, estimated_noise_level)
                elif hybrid_flag == 1:
                    if self.residual:
                        pred_imgs = pred_imgs + self.second_stage(blur_img, sharp_imgs, estimated_noise_level)
                    else:
                        pred_imgs = self.second_stage(blur_img, sharp_imgs, estimated_noise_level)
                else:
                    raise ValueError
            else:
                if self.residual:
                    pred_imgs = pred_imgs + self.second_stage(blur_img, pred_imgs, estimated_noise_level)
                else:
                    pred_imgs = self.second_stage(blur_img, pred_imgs, estimated_noise_level)
            all_pred_imgs.append(pred_imgs)

        # Calculate losses
        loss = 0.
        for pred_imgs in all_pred_imgs:
            # Reconstruction loss
            loss += self.l2(pred_imgs, sharp_imgs)
            if self.perc_ratio > 0:
                b, num_gts, c, h, w = pred_imgs.shape
                # Perceptual loss
                x_vgg = self.vgg(pred_imgs.reshape(b * num_gts, c, h, w))
                y_vgg = self.vgg(sharp_imgs.reshape(b * num_gts, c, h, w))
                for i in range(len(x_vgg)):
                    loss += self.perc_ratio * torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()

        # NEW: Add noise estimation loss
        if self.use_noise_aware and gt_noise_level is not None:
            loss += self.noise_loss_weight * noise_estimation_loss
            out_tensor['noise_est_loss'] = noise_estimation_loss.detach()

        # Backward propagation
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Register tensors
        out_tensor['loss'] = loss
        out_tensor['inp_img'] = self.val_range * blur_img[:, :c].detach()
        out_tensor['trend_img'] = trend_img.detach()
        out_tensor['pred_imgs'] = torch.clamp(self.val_range * all_pred_imgs[-1].detach(), min=0,
                                              max=self.val_range)
        out_tensor['gt_imgs'] = self.val_range * sharp_imgs

        return out_tensor

    def first_stage(self, blur_img, trend_img, noise_level=None):
        """
        :param blur_img: blurry image (b, c, h, w), c = 3
        :param trend_img: trend guidance (b, tc, h, w), tc = 2
        :param noise_level: estimated noise level (b, 1)
        :return: sharp image sequence (b, num_gts, c, h, w)
        """
        # NEW: Set noise level for FiLM modulation
        if self.use_noise_aware and hasattr(self.decomposer_s1, 'module'):
            # DDP wrapped
            self.decomposer_s1.module.set_noise_level(noise_level)
        elif self.use_noise_aware and hasattr(self.decomposer_s1, 'set_noise_level'):
            # Not DDP wrapped
            self.decomposer_s1.set_noise_level(noise_level)

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

    def second_stage(self, blur_img, pred_imgs, noise_level=None):
        """
        :param blur_img: blurry image (b, c, h, w)
        :param pred_imgs: predicted images (b, num_gts, c, h, w)
        :param noise_level: estimated noise level (b, 1)
        :return: predicted images after refinement (b, num_gts, c, h, w)
        """
        # NEW: Set noise level for FiLM modulation
        if self.use_noise_aware and hasattr(self.decomposer_s2, 'module'):
            # DDP wrapped
            self.decomposer_s2.module.set_noise_level(noise_level)
        elif self.use_noise_aware and hasattr(self.decomposer_s2, 'set_noise_level'):
            # Not DDP wrapped
            self.decomposer_s2.set_noise_level(noise_level)

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
        """
        :param blur_img: blurry image (h, w, 3), 0~255, RGB
        :param trend: flow trend guidance (2, ) or (h, w, 2)
        :return: predicted images (num_gts, h, w, 3)
        """
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

        # NEW: Estimate noise level during inference
        estimated_noise_level = None
        if self.use_noise_aware:
            estimated_noise_level = self.noise_estimator(blur_img)

        pred_imgs_s1 = self.first_stage(blur_img=blur_img, trend_img=trend, noise_level=estimated_noise_level)
        torch.cuda.synchronize()
        for i in range(num_iters):
            if self.residual:
                pred_residual = self.second_stage(blur_img=blur_img, pred_imgs=pred_imgs_s1,
                                                  noise_level=estimated_noise_level)
                pred_imgs_s2 = pred_imgs_s1 + pred_residual
            else:
                pred_imgs_s2 = self.second_stage(blur_img=blur_img, pred_imgs=pred_imgs_s1,
                                                 noise_level=estimated_noise_level)
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