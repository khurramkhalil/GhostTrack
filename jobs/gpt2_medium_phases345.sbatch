#!/bin/bash
#SBATCH --job-name=gpt2m-phases345
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# ==========================================
# GhostTrack Phases 3-5 - GPT-2 Medium
# ==========================================
# Runs tracking, detection, and evaluation
# Requires Phase 1&2 to be completed (SAEs trained)

echo "================================================================"
echo "GhostTrack GPT-2 Medium - Phases 3-5"
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

# Configuration
MODEL_CONFIG="./config/model_configs/gpt2-medium.yaml"

echo ""
echo "Configuration:"
echo "  Model: GPT-2 Medium (24 layers, 1024 dim)"
echo "  SAE Dir: ./models/checkpoints/gpt2-medium"
echo "  Results: ./results/gpt2-medium"
echo ""

# Verify SAEs exist
echo "=== Verifying SAE checkpoints ==="
SAE_COUNT=$(ls -1 ./models/checkpoints/gpt2-medium/sae_layer_*_best.pt 2>/dev/null | wc -l)
echo "Found $SAE_COUNT SAE checkpoints"

if [ $SAE_COUNT -lt 24 ]; then
    echo "ERROR: Expected 24 SAEs, found $SAE_COUNT. Phases 1-2 may not be complete."
    exit 1
fi

# ==========================================
# PHASE 3: Hypothesis Tracking
# ==========================================
echo ""
echo "================================================================"
echo "PHASE 3: HYPOTHESIS TRACKING"
echo "================================================================"
echo ""

python scripts/run_tracking.py \
    --config ${MODEL_CONFIG} \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir ./results/gpt2-medium/tracking \
    --device cuda \
    --num-samples 500

TRACK_STATUS=$?

echo ""
echo "Tracking phase completed with status: $TRACK_STATUS"

if [ $TRACK_STATUS -ne 0 ]; then
    echo "ERROR: Tracking failed! Check logs."
    exit $TRACK_STATUS
fi

# ==========================================
# PHASE 4: Hallucination Detection
# ==========================================
echo ""
echo "================================================================"
echo "PHASE 4: HALLUCINATION DETECTION"
echo "================================================================"
echo ""

python scripts/run_detection.py \
    --config ${MODEL_CONFIG} \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir ./results/gpt2-medium/tracking \
    --output-dir ./results/gpt2-medium/detection \
    --device cuda

DETECT_STATUS=$?

echo ""
echo "Detection phase completed with status: $DETECT_STATUS"

if [ $DETECT_STATUS -ne 0 ]; then
    echo "ERROR: Detection failed! Check logs."
    exit $DETECT_STATUS
fi

# ==========================================
# PHASE 5: Evaluation & Metrics
# ==========================================
echo ""
echo "================================================================"
echo "PHASE 5: EVALUATION & METRICS"
echo "================================================================"
echo ""

python scripts/run_evaluation.py \
    --config ${MODEL_CONFIG} \
    --results-dir ./results/gpt2-medium \
    --output-dir ./results/gpt2-medium/evaluation

EVAL_STATUS=$?

echo ""
echo "Evaluation phase completed with status: $EVAL_STATUS"

# ==========================================
# FINAL SUMMARY
# ==========================================
echo ""
echo "================================================================"
echo "PHASES 3-5 COMPLETE"
echo "================================================================"
echo "Job completed at $(date)"
echo ""
echo "Status Summary:"
echo "  Phase 3 (Tracking):    $TRACK_STATUS"
echo "  Phase 4 (Detection):   $DETECT_STATUS"
echo "  Phase 5 (Evaluation):  $EVAL_STATUS"
echo ""
echo "Results saved to:"
ls -lh ./results/gpt2-medium/

echo ""
echo "=== Final Report ==="
cat ./results/gpt2-medium/evaluation/EVALUATION_REPORT.md 2>/dev/null || echo "Report not generated"

echo ""
echo "Total job duration: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
