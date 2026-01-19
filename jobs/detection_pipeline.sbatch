#!/bin/bash
#SBATCH --job-name=ghosttrack-detect
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# ==========================================
# GhostTrack Phase 3 & 4 - Detection Pipeline
# ==========================================
# Runs Hypothesis Tracking on TruthfulQA
# Trains and Evaluates Hallucination Detector

echo "================================================================"
echo "GhostTrack Detection Pipeline"
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

# Run Detection Pipeline
echo ""
echo "Starting Detection Pipeline..."
export PYTHONPATH=$PYTHONPATH:.
python scripts/run_detection_pipeline.py \
    --checkpoint-dir ./models/checkpoints \
    --output-dir ./results/detection \
    --device cuda \
    --seed 42

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Pipeline completed at $(date)"
echo "Exit status: $EXIT_CODE"
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Results:"
    ls -la ./results/detection/
    echo ""
    cat ./results/detection/results.json
fi
