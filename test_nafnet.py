import torch
from NAFNet.basicsr.models import create_model
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img, imwrite
import os

# Model configuration
opt = {
    'model_type': 'NAFNet',
    'img_channel': 3,
    'width': 64,
    'middle_blk_num': 1,
    'enc_blk_nums': [1, 1, 1, 1],
    'dec_blk_nums': [1, 1, 1, 1],
    'pretrained_path': 'NAFNet/pretrained_weights/NAFNet-GoPro-width64.pth',
    'device': 'cuda:0'
}

# Load model
denoiser = create_model(opt)
checkpoint = torch.load(opt['pretrained_path'], map_location=opt['device'])
denoiser.load_state_dict(checkpoint['params'], strict=False)
denoiser = denoiser.to(opt['device']).eval()
print("NAFNet model loaded")

# Paths
input_path = './NAFNet/demo/noisy.png'
output_path = './NAFNet/demo/denoise_img.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Read image
file_client = FileClient('disk')
img_bytes = file_client.get(input_path, None)
img = imfrombytes(img_bytes, float32=True)
img = img2tensor(img, bgr2rgb=True, float32=True)
img = img.unsqueeze(0).to(opt['device'])

# Run inference
with torch.no_grad():
    denoised = denoiser(img)

# Save output
denoised_img = tensor2img([denoised])
imwrite(denoised_img, output_path)
print(f"Denoised image saved to {output_path}")
