#!/bin/bash
#SBATCH --job-name=check
#SBATCH --time=00:02:00
#SBATCH --output=logs/%j.out

cd ~/unofficail_audio_deepfake/D3
python check.py
