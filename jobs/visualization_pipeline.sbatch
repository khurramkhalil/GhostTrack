#!/bin/bash
#SBATCH --job-name=ghosttrack-viz
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# ==========================================
# GhostTrack Phase 5 - Visualization Pipeline
# ==========================================
# Generates interactive dashboards and case studies
# using trained SAEs and Detector.

echo "================================================================"
echo "GhostTrack Visualization Pipeline"
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# Load modules and activate environment
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek

# Navigate to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

# Run Visualization Script
echo ""
echo "Starting Visualization..."
export PYTHONPATH=$PYTHONPATH:.
python scripts/run_visualization.py \
    --checkpoint-dir ./models/checkpoints \
    --detector-path ./results/detection/detector.pkl \
    --output-dir ./results/visualization \
    --device cuda \
    --num-examples 10

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Pipeline completed at $(date)"
echo "Exit status: $EXIT_CODE"
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Results:"
    ls -la ./results/visualization/
fi
