#!/bin/bash
#SBATCH --job-name=qwen-train-p3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Load modules
module load miniconda3/4.10.3_gcc_9.5.0
module load cuda/11.8

# Activate environment
source activate deepseek

# Change to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

CONFIG="config/model_configs/qwen-1.5b.yaml"

echo "================================================================"
echo "Starting Qwen Training Part 3 (Layers 19-27)"
echo "Config: $CONFIG"
echo "================================================================"

python3 scripts/extract_and_train.py \
    --config $CONFIG \
    --device cuda \
    --train-only \
    --layers 19,20,21,22,23,24,25,26,27 \
    --save-dir ./models/checkpoints/qwen \
    --cache-dir ./data/cache/qwen

echo "Part 3 Complete!"
