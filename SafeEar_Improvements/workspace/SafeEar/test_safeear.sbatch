#!/bin/bash
#SBATCH --job-name=safear_test
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_%j.out

source ~/.bashrc
conda activate mysafear

cd ~/unofficail_audio_deepfake/D3/SafeEar

python test.py \
  --conf_dir config/train21.yaml \
  --ckpt_path Exps/ASVspoof21/checkpoints/epoch=18-val_eer=0.0381.ckpt
