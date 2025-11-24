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
    
    for frame_idx, frame in enumerate(frames):
        img = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        filename = f'{video_name}_sample{sample_idx:03d}_frame{frame_idx:03d}.png'
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img_bgr)
    
    print(f'Saved {len(frames)} frames for {video_name} sample {sample_idx}')

dist.init_process_group(backend="nccl")
torch.cuda.set_device(0)

config_path = '/w/20251/mahikav/experiments/gopro_mbd_mahika/cfg.yaml'
with open(config_path) as f:
    configs = yaml.full_load(f)

configs['resume_dir'] = '/w/20251/mahikav/experiments/gopro_mbd_mahika'
configs['dataset_args']['root_dir'] = ['/w/20251/mahikav/Gopro']
configs['dataset_args']['use_trend'] = False

num_gts = configs['dataset_args']['num_gts']
print(f'Model generates {num_gts} frames per input')

model = MBD(local_rank=0, configs=configs)
test_dataset = BDDataset(set_type='valid', **configs['dataset_args'])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

output_dir = 'generated_sequences_one_per_video'
os.makedirs(output_dir, exist_ok=True)

# One sample from each of the 11 test videos
sample_indices = [0, 20, 40, 94, 114, 134, 154, 174, 194, 214, 220]

with torch.no_grad():
    for i, tensor in enumerate(test_loader):
        if i not in sample_indices:
            continue
            
        tensor['inp'] = tensor['inp'].cuda()
        tensor['gt'] = tensor['gt'].cuda()
        video_name = tensor['video'][0] if isinstance(tensor['video'], list) else tensor['video']
        
        print(f'Processing sample {i} from video: {video_name}')
        
        out_tensor = model.update(inp_tensor=tensor, training=False)
        pred_frames = out_tensor['pred_imgs'][0]
        
        save_frame_sequence(pred_frames, output_dir, video_name, i)

dist.destroy_process_group()
print(f'\nDone! Generated 7 frames each from {len(sample_indices)} different videos')
print(f'Total: {len(sample_indices) * 7} output frames in {output_dir}/')
