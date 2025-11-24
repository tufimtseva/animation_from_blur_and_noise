import torch
import torch.distributed as dist
import yaml
import cv2
import numpy as np
import os
from model.MBD import MBD
from data.dataset import BAistPP as BDDataset
from torch.utils.data import DataLoader

def save_frame_sequence(frames, save_dir, video_name, sample_idx):
    """Save all predicted frames as separate images"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Debug: check if frames are different
    print(f"  Frame tensor shape: {frames.shape}")
    
    for frame_idx, frame in enumerate(frames):
        img = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        filename = f'{video_name}_sample{sample_idx:03d}_frame{frame_idx:03d}.png'
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img_bgr)
        
        # Debug: print some pixel values to verify frames differ
        if frame_idx == 0:
            print(f"  Frame 0 first pixel: {img[0,0,:]}")
        elif frame_idx == 6:
            print(f"  Frame 6 first pixel: {img[0,0,:]}")
    
    print(f'Saved {len(frames)} frames')

dist.init_process_group(backend="nccl")
torch.cuda.set_device(0)

config_path = '/w/20251/mahikav/experiments/gopro_mbd_mahika/cfg.yaml'
with open(config_path) as f:
    configs = yaml.full_load(f)

configs['resume_dir'] = '/w/20251/mahikav/experiments/gopro_mbd_mahika'
configs['dataset_args']['root_dir'] = ['/w/20251/mahikav/Gopro']
configs['dataset_args']['use_trend'] = False

num_gts = configs['dataset_args']['num_gts']

model = MBD(local_rank=0, configs=configs)
test_dataset = BDDataset(set_type='valid', **configs['dataset_args'])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

output_dir = 'generated_sequences_debug'
os.makedirs(output_dir, exist_ok=True)

# Just test with first sample
sample_indices = [0]

with torch.no_grad():
    for i, tensor in enumerate(test_loader):
        if i not in sample_indices:
            continue
            
        tensor['inp'] = tensor['inp'].cuda()
        tensor['gt'] = tensor['gt'].cuda()
        video_name = tensor['video'][0] if isinstance(tensor['video'], list) else tensor['video']
        
        print(f'\nProcessing sample {i} from video: {video_name}')
        
        out_tensor = model.update(inp_tensor=tensor, training=False)
        
        # Debug: check output tensor structure
        print(f"Output keys: {out_tensor.keys()}")
        print(f"pred_imgs shape: {out_tensor['pred_imgs'].shape}")
        
        pred_frames = out_tensor['pred_imgs'][0]  # Should be (7, 3, H, W)
        
        save_frame_sequence(pred_frames, output_dir, video_name, i)

dist.destroy_process_group()
