#!/bin/bash
#SBATCH --error=slurm-%j.err
#SBATCH --job-name=anim_eval
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=33G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=gpunode16

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL


source ~/miniconda3/etc/profile.d/conda.sh
conda activate animation-from-blur
cd ~/animation_from_blur_and_noise

torchrun --nproc_per_node=1 train_mbd.py \
  --config ./configs/cfg.yaml \
  --log_dir ./log \
  --verbose

#torchrun  train_mbd.py \
#--predictor_resume_dir ./checkpoints/baistpp_cvaegan_gumble_celoss_z4/ \
#--decomposer_resume_dir ./checkpoints/baistpp_mbd_2s_residual/ \
#--data_dir /w/20251/shreyashree/Gopro \
#-ns 1 \