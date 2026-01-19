#!/bin/bash
#SBATCH --job-name=ghosttrack-ablations
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# ==========================================
# GhostTrack Phase 6 - Ablation Studies
# ==========================================
# Runs 4 ablation experiments to validate methodology:
# 1. Baseline
# 2. Single Hypothesis (Top-1)
# 3. No Association (Independent Layers)
# 4. Feature ID Association (Random Baseline)

echo "================================================================"
echo "GhostTrack Ablation Studies"
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# Load modules and activate environment
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek

# Navigate to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

# Check GPU availability
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run Ablations
echo ""
echo "Starting Ablation Studies..."
export PYTHONPATH=$PYTHONPATH:.
python scripts/run_ablations.py \
    --checkpoint-dir ./models/checkpoints \
    --output-dir ./results/ablations \
    --device cuda \
    --seed 42

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Ablations completed at $(date)"
echo "Exit status: $EXIT_CODE"
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Summary:"
    ls -la ./results/ablations/
    if [ -f ./results/ablations/ablation_results.json ]; then
        echo ""
        echo "Results Preview:"
        head -n 20 ./results/ablations/ablation_results.json
    fi
fi
