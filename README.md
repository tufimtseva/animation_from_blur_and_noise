# Motion Reconstruction from a Single Blurry Noisy Image using Deep Image Deblurring and Sequence Modeling

This repository contains our re-implementation and extension of the Animation from Blur framework with a new noise-aware decomposition module, improved training pipeline, and evaluation framework. Our goal is to reconstruct motion trajectories and generate plausible video sequences from a single motion-blurred and potentially noisy image.
We introduce:

* A Noise-Aware Motion Blur Decomposition (NAMBD) module  
* A noise estimator network that predicts σ  
* FiLM-based feature modulation to adaptively condition features on noise level  
* A Restormer-based pre-denoising option for extreme noise conditions  
* A temporal consistency loss to improve metrics  

## Installation
We recommend using conda.

1. Create the environment
```
conda create -n animation-from-blur python=3.8
conda activate animation-from-blur
```
2. Install Python dependencies
```
pip install -r requirements.txt
```

## Dataset Preparation
This project uses the GoPro dataset: https://seungjunnah.github.io/Datasets/gopro.html <br> 
Expected directory structure: <br> 
```
Gopro/
  train/
  test/
```
You may place this directory anywhere, but when running scripts, use:
```
--data_dir /path/to/Gopro
```

## Pretrained Weights
Google Drive download link: 
You may store checkpoints anywhere.
Pass their paths using ```--predictor_resume_dir``` and ```--decomposer_resume_dir```

## Training
```
torchrun --nproc_per_node=1 train_mbd.py \
  --config ./configs/cfg.yaml \
  --log_dir ./logs \
  --verbose
```

## Evaluation
```
torchrun valid_vaegan.py \
  --predictor_resume_dir ./checkpoints/predictor_checkpoint/ \
  --decomposer_resume_dir ./checkpoints/decomposer_checkpoint/ \
  --data_dir /path/to/Gopro \
  -ns 1 \
  --verbose

```
## Citations
#### Original Animation from Blur paper
```
@inproceedings{zhong2022animation,
  title={Animation from blur: Multi-modal blur decomposition with motion guidance},
  author={Zhong, Zhihang and Sun, Xiao and Wu, Zhirong and Zheng, Yinqiang and Lin, Stephen and Sato, Imari},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XIX},
  pages={599--615},
  year={2022},
  organization={Springer}
}
@inproceedings{zhong2023blur,
  title={Blur Interpolation Transformer for Real-World Motion from Blur},
  author={Zhong, Zhihang and Cao, Mingdeng and Ji, Xiang and Zheng, Yinqiang and Sato, Imari},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5713--5723},
  year={2023}
}
```
