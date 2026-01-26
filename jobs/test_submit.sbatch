#!/bin/bash
#SBATCH --job-name=test-submit
#SBATCH --output=%x_%j.out
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

echo "Can I run?"
