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

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Activate environment
conda activate animation-from-blur

# Evaluation script
torchrun  valid_vaegan.py \
  --predictor_resume_dir ./checkpoints/baistpp_cvaegan_gumble_celoss_z4/ \
  --decomposer_resume_dir /w/20251/shreyashree/log \
  --data_dir /w/20251/shreyashree/Gopro \
  -ns 1 \
  --verbose