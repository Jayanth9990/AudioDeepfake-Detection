#!/bin/bash
#SBATCH --job-name=apcodec_train
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

#SBATCH --output=apcodec.out
#SBATCH --error=apcodec.err

# Load conda properly (CORRECT PATH)
source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.10-htq3o45qwkhxfedgb65w6kbeb2227u7r/etc/profile.d/conda.sh

# Activate your environment
conda activate /scratch/sameera/envs/apcodec

# Debug (important)
echo "Python path:"
which python

echo "GPU status:"
nvidia-smi

# Go to project
cd /scratch/sameera/jay/APCodec

# Run training
python train.py
