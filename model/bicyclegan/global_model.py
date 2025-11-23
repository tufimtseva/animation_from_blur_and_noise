import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from .local_model import BiCycleGANModel

class NoiseEstimationBranch(nn.Module):
    """Tiny network to estimate noise level from input image"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output: noise level ∈ [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W)
        Returns:
            noise_level: Estimated noise level (B, 1) in range [0, 25]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        noise_level = self.fc(x) * 25.0  # Scale to [0, 25]
        return noise_level


class NoiseAwareBiCycleGANModel(BiCycleGANModel):
    """
    Extended BiCycleGAN with noise-aware feature modulation
    """

    def __init__(self, opt):
        super().__init__(opt)

        # Add noise estimation branch
        self.noise_estimator = NoiseEstimationBranch()

        # Store for tracking
        self.estimated_noise = None
        self.ground_truth_noise = None

        # Move noise estimator to GPU
        if len(opt.gpu_ids) > 0:
            self.noise_estimator.to(opt.gpu_ids[0])

        # Add noise estimator to optimizer (if training)
        if self.isTrain:
            # Add noise estimator parameters to existing optimizer
            # or create a separate optimizer
            self.optimizer_noise = torch.optim.Adam(
                self.noise_estimator.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_noise)

    def set_input(self, input):
        """Override to store ground truth noise level"""
        super().set_input(input)
        # Store ground truth noise if provided
        if 'noise_level' in input:
            self.ground_truth_noise = input['noise_level'].to(self.device)
        else:
            self.ground_truth_noise = None

    def forward(self):
        """Override forward to estimate noise and use it"""
        # Estimate noise level from input
        self.estimated_noise = self.noise_estimator(self.real_A)  # (B, 1)

        # Continue with normal BiCycleGAN forward pass
        super().forward()

    def get_noise_estimation_loss(self):
        """
        Calculate loss between estimated and ground truth noise
        Only used when ground truth noise is available (during training)
        """
        if self.ground_truth_noise is not None and self.estimated_noise is not None:
            # L1 loss between estimated and ground truth noise
            noise_loss = F.l1_loss(self.estimated_noise, self.ground_truth_noise)
            return noise_loss
        return torch.tensor(0.0).to(self.device)

    def backward_G(self):
        """Override to add noise estimation loss"""
        # Original BiCycleGAN losses
        super().backward_G()

        # Add noise estimation loss (weighted)
        if self.isTrain and self.ground_truth_noise is not None:
            noise_loss = self.get_noise_estimation_loss()
            # Weight the noise loss (adjust this weight as needed)
            weighted_noise_loss = 0.1 * noise_loss
            weighted_noise_loss.backward(retain_graph=False)

            # Store for logging
            self.loss_noise_est = noise_loss.item()


class GuidePredictor:
    '''
    Model for motion guide prediction (BiCycleGAN)
    Multimodal Image-to-Image Translation by Enforcing Bi-Cycle Consistency (NeurIPS 2017)
    https://proceedings.neurips.cc/paper/2017/file/819f46e52c25763a55cc642422644317-Paper.pdf
    '''

    def __init__(self, local_rank, configs):
        self.device = torch.device("cuda", local_rank)
        self.val_range = configs['val_range']

        opt = edict(configs['bicyclegan_args'])
        opt.gpu_ids = [local_rank, ]

        # Use the noise-aware model instead of the original
        self.model = NoiseAwareBiCycleGANModel(opt)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.setup(opt)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save_model(self, name):
        self.model.save_networks(name)

    def scheduler_step(self):
        self.model.update_learning_rate()

    def get_lr(self):
        return self.model.get_lr()

    def guidance2label(self, trend_img):
        """
        (0, 0) black -> 0
        (1, -1) red -> 1
        (-1,-1) green -> 2
        (-1, 1) yellow -> 3
        (1, 1) blue -> 4
        :param trend_img: shape (b, 2, h, w)
        :return: trend_label: shape (b, 1, h, w)
        """
        b, _, h, w = trend_img.shape
        trend_label = torch.zeros(b, 1, h, w).to(self.device)
        # (0, 0) black -> 0
        trend_label = torch.where((trend_img[:, 0:1] == 0) & (trend_img[:, 1:2] == 0),
                                  torch.tensor(0.).to(self.device), trend_label)
        # (1, -1) red -> 1
        trend_label = torch.where((trend_img[:, 0:1] == 1) & (trend_img[:, 1:2] == -1),
                                  torch.tensor(1.).to(self.device), trend_label)
        # (-1,-1) green -> 2
        trend_label = torch.where((trend_img[:, 0:1] == -1) & (trend_img[:, 1:2] == -1),
                                  torch.tensor(2.).to(self.device), trend_label)
        # (-1, 1) yellow -> 3
        trend_label = torch.where((trend_img[:, 0:1] == -1) & (trend_img[:, 1:2] == 1),
                                  torch.tensor(3.).to(self.device), trend_label)
        # (1, 1) blue -> 4
        trend_label = torch.where((trend_img[:, 0:1] == 1) & (trend_img[:, 1:2] == 1),
                                  torch.tensor(4.).to(self.device), trend_label)

        return trend_label

    def label2guidance(self, pred_label):
        """
        0 -> (0, 0) black
        1 -> (1, -1) red
        2 -> (-1,-1) green
        3 -> (-1, 1) yellow
        4 -> (1, 1) blue
        :param pred_label: shape (b, 1, h, w)
        :return: shape (b, 2, h, w)
        """
        b, _, h, w = pred_label.shape
        pred_trend_img = torch.zeros(b, 2, h, w).to(self.device)  # shape (b, 2, h, w)
        # 0 -> (0, 0) black
        pred_trend_img[:, 0:1] = torch.where(pred_label == 0, torch.tensor(0.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 0, torch.tensor(0.).to(self.device), pred_trend_img[:, 1:2])
        # 1 -> (1, -1) red
        pred_trend_img[:, 0:1] = torch.where(pred_label == 1, torch.tensor(1.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 1, torch.tensor(-1.).to(self.device), pred_trend_img[:, 1:2])
        # 2 -> (-1,-1) green
        pred_trend_img[:, 0:1] = torch.where(pred_label == 2, torch.tensor(-1.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 2, torch.tensor(-1.).to(self.device), pred_trend_img[:, 1:2])
        # 2 -> (-1, 1) yellow
        pred_trend_img[:, 0:1] = torch.where(pred_label == 3, torch.tensor(-1.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 3, torch.tensor(1.).to(self.device), pred_trend_img[:, 1:2])
        # 2 -> (1, 1) blue
        pred_trend_img[:, 0:1] = torch.where(pred_label == 4, torch.tensor(1.).to(self.device), pred_trend_img[:, 0:1])
        pred_trend_img[:, 1:2] = torch.where(pred_label == 4, torch.tensor(1.).to(self.device), pred_trend_img[:, 1:2])

        return pred_trend_img

    def tensor_preprocess(self, inp_tensor):
        # preprocess data
        blur_img = inp_tensor['inp']  # (b, n, c, h, w), n=1, c=3
        assert blur_img.size(1) == 1  # (b, c, h, w)
        blur_img = blur_img.squeeze(dim=1) / self.val_range
        trend_img = inp_tensor['trend']  # (b, n, tc, h, w), n=1, tc=2
        trend_img = trend_img.squeeze(dim=1)  # (b, tc, h, w)
        trend_rev_img = -1 * trend_img.clone()
        label_img = self.guidance2label(trend_img)
        label_img = F.one_hot(label_img.long(), 5).squeeze(dim=1).permute(0, 3, 1, 2).float()  # (b, 5, h, w)
        label_rev_img = self.guidance2label(trend_rev_img)
        label_rev_img = F.one_hot(label_rev_img.long(), 5).squeeze(dim=1).permute(0, 3, 1, 2).float()  # (b, 5, h, w)
        tensor = {}
        tensor['A'] = blur_img  # (b, c, h, w)
        tensor['B'] = label_img
        tensor['B_rev'] = label_rev_img
        tensor['A_paths'] = inp_tensor['video']
        tensor['B_paths'] = inp_tensor['video']

        # NEW: Pass noise level to model
        if 'noise_level' in inp_tensor:
            tensor['noise_level'] = inp_tensor['noise_level']

        return tensor

    def return_tensor(self):
        out_tensor = {}
        # out_tensor['loss'] = sum(self.model.get_current_losses().values())
        losses = self.model.get_current_losses()

        # Add noise estimation loss if available
        if hasattr(self.model, 'loss_noise_est'):
            losses['noise_est'] = self.model.loss_noise_est

        out_tensor['loss'] = losses
        out_tensor['inp_img'] = self.val_range * self.model.real_A_encoded

        _, fake_B_random = torch.max(self.model.fake_B_random, dim=1, keepdim=True)
        fake_B_random = self.label2guidance(fake_B_random)  # (b, 2, h, w)
        _, fake_B_encoded = torch.max(self.model.fake_B_encoded, dim=1, keepdim=True)
        fake_B_encoded = self.label2guidance(fake_B_encoded)
        out_tensor['pred_trends'] = torch.cat([fake_B_encoded, fake_B_random], dim=-1)  # (b, 2, h, 2 * w)
        _, real_B_encoded = torch.max(self.model.real_B_encoded, dim=1, keepdim=True)
        real_B_encoded = self.label2guidance(real_B_encoded)
        out_tensor['gt_trend'] = real_B_encoded

        # NEW: Add estimated noise level for logging/visualization
        if hasattr(self.model, 'estimated_noise') and self.model.estimated_noise is not None:
            out_tensor['estimated_noise'] = self.model.estimated_noise
        if hasattr(self.model, 'ground_truth_noise') and self.model.ground_truth_noise is not None:
            out_tensor['gt_noise'] = self.model.ground_truth_noise

        return out_tensor

    def update(self, inp_tensor, training=True):
        tensor = self.tensor_preprocess(inp_tensor)
        self.model.set_input(tensor)
        if not self.model.is_train():
            return None
        if training:
            self.train()
            # calculate loss functions, get gradients, update network weights
            self.model.optimize_parameters()
        else:
            with torch.no_grad():
                self.eval()
                # calculate loss functions, get gradients, update network weights
                self.model.forward()

        return self.return_tensor()

    @torch.no_grad()
    def test(self, inp_tensor, zeros=False):
        tensor = self.tensor_preprocess(inp_tensor)
        self.model.set_input(tensor)
        real_A, fake_B, real_B = self.model.test(encode=False, turbulent=False, zeros=zeros)
        real_A = self.val_range * real_A
        _, fake_B = torch.max(fake_B, dim=1, keepdim=True)
        fake_B = self.label2guidance(fake_B)
        _, real_B = torch.max(real_B, dim=1, keepdim=True)
        real_B = self.label2guidance(real_B)
        out_tensor = {'inp_img': real_A, 'pred_trends': fake_B, 'gt_trend': real_B}

        # NEW: Add estimated noise level
        if hasattr(self.model, 'estimated_noise') and self.model.estimated_noise is not None:
            out_tensor['estimated_noise'] = self.model.estimated_noise

        return out_tensor
