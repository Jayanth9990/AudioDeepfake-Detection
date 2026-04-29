#!/bin/bash
#SBATCH --job-name=safeear
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- W&B: optional ---
unset WANDB_MODE
export WANDB_MODE=offline
export WANDB_SILENT=true

# --- Environment ---
source ~/.bashrc
conda activate mysafear

# --- Project root ---
cd ~/unofficail_audio_deepfake/D3/SafeEar

# ensure python can find 'safeear' package
export PYTHONPATH=$PWD

# --- Run (Hydra correct syntax) ---
python train.py --config-path config --config-name train21.yaml
