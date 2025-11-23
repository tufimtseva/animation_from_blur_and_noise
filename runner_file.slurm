#!/bin/bash
#SBATCH --error=slurm-%j.err
#SBATCH --job-name=anim_eval
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=33G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out  # log file
#SBATCH --exclude=gpunode16

# Load conda (adjust if needed)
source ~/miniconda3/etc/profile.d/conda.sh

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Activate environment (must already exist!)
conda activate animation-from-blur

# Move to project directory


# Run the evaluation script
torchrun  valid_vaegan.py \
  --predictor_resume_dir ./checkpoints/baistpp_cvaegan_gumble_celoss_z4/ \
  --decomposer_resume_dir ./checkpoints/baistpp_mbd_2s_residual/ \
  --data_dir /w/20251/shreyashree/Gopro \
  -ns 1 \
  --verbose