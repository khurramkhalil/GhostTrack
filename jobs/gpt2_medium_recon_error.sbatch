#!/bin/bash
#SBATCH --job-name=gpt2m-recon
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00

# ==========================================
# GhostTrack Enhanced - Reconstruction Error Features
# ==========================================

echo "================================================================"
echo "GhostTrack GPT-2 Medium - Enhanced with Reconstruction Error"
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# Load environment
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek
cd /cluster/VAST/hoquek-lab/GhostTrack

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

BASE_CONFIG="./config/model_configs/gpt2-medium.yaml"
RESULTS_BASE="./results/gpt2-medium/recon_error_exp"
NUM_SAMPLES=300

mkdir -p $RESULTS_BASE

echo ""
echo "--- Running Enhanced Tracking with Reconstruction Error ---"

python scripts/run_tracking_v2.py \
    --config $BASE_CONFIG \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir $RESULTS_BASE/tracking \
    --num-samples $NUM_SAMPLES

echo ""
echo "--- Running Detection on Enhanced Features ---"

python scripts/run_detection.py \
    --config $BASE_CONFIG \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir $RESULTS_BASE/tracking \
    --output-dir $RESULTS_BASE/detection

echo ""
echo "================================================================"
echo "EXPERIMENT COMPLETE"
echo "================================================================"

if [ -f "$RESULTS_BASE/detection/detection_metrics.json" ]; then
    echo "Results:"
    cat $RESULTS_BASE/detection/detection_metrics.json
else
    echo "Metrics not found"
fi

echo ""
echo "Job completed at $(date)"
