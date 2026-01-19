#!/bin/bash
#SBATCH --job-name=ghosttrack-extract
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# ==========================================
# GhostTrack Phase 1: Extract Hidden States
# ==========================================

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules and activate environment
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek

# Navigate to project directory (VAST storage)
cd /cluster/VAST/hoquek-lab/GhostTrack

# Check GPU availability
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run extraction for all layers
python scripts/extract_and_train.py \
    --extract-only \
    --layers all \
    --num-tokens 10000000 \
    --batch-size-extract 16 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints

echo "Extraction completed at $(date)"
