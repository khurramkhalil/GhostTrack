#!/bin/bash
#SBATCH --job-name=gpt2m-full
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=168:00:00

# ==========================================
# GhostTrack Full Pipeline - GPT-2 Medium
# ==========================================
# Complete end-to-end pipeline for GPT-2 Medium (24 layers)
# Running on 4x A100 GPUs with optimized batch sizes
# 
# Pipeline Phases:
# 1. Hidden State Extraction (parallel across 4 GPUs)
# 2. SAE Training (parallel across 4 GPUs)
# 3. Hypothesis Tracking
# 4. Hallucination Detection
# 5. Evaluation & Metrics

echo "================================================================"
echo "GhostTrack GPT-2 Medium - Full Pipeline"
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
NUM_TOKENS=10000000
EXTRACT_BATCH_SIZE=16  # Smaller for larger model - avoid OOM
TRAIN_BATCH_SIZE=128   # Optimized for A100 - good utilization without OOM

echo ""
echo "Configuration:"
echo "  Model: GPT-2 Medium (24 layers, 1024 dim)"
echo "  Tokens per layer: ${NUM_TOKENS}"
echo "  Extraction batch size: ${EXTRACT_BATCH_SIZE}"
echo "  Training batch size: ${TRAIN_BATCH_SIZE}"
echo ""

# ==========================================
# PHASE 1: Parallel Hidden State Extraction
# ==========================================
echo "================================================================"
echo "PHASE 1: EXTRACTING HIDDEN STATES (24 layers, 4 GPUs)"
echo "================================================================"
echo "GPU allocation: 6 layers per GPU"
echo ""

# GPU 0: Layers 0-5
CUDA_VISIBLE_DEVICES=0 python scripts/extract_and_train.py \
    --extract-only \
    --layers 0,1,2,3,4,5 \
    --num-tokens ${NUM_TOKENS} \
    --batch-size-extract ${EXTRACT_BATCH_SIZE} \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-medium \
    --save-dir ./models/checkpoints/gpt2-medium &
PID0=$!

# GPU 1: Layers 6-11
CUDA_VISIBLE_DEVICES=1 python scripts/extract_and_train.py \
    --extract-only \
    --layers 6,7,8,9,10,11 \
    --num-tokens ${NUM_TOKENS} \
    --batch-size-extract ${EXTRACT_BATCH_SIZE} \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-medium \
    --save-dir ./models/checkpoints/gpt2-medium &
PID1=$!

# GPU 2: Layers 12-17
CUDA_VISIBLE_DEVICES=2 python scripts/extract_and_train.py \
    --extract-only \
    --layers 12,13,14,15,16,17 \
    --num-tokens ${NUM_TOKENS} \
    --batch-size-extract ${EXTRACT_BATCH_SIZE} \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-medium \
    --save-dir ./models/checkpoints/gpt2-medium &
PID2=$!

# GPU 3: Layers 18-23
CUDA_VISIBLE_DEVICES=3 python scripts/extract_and_train.py \
    --extract-only \
    --layers 18,19,20,21,22,23 \
    --num-tokens ${NUM_TOKENS} \
    --batch-size-extract ${EXTRACT_BATCH_SIZE} \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-medium \
    --save-dir ./models/checkpoints/gpt2-medium &
PID3=$!

echo "Extraction PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Waiting for all extraction jobs to complete..."

# Wait for all extraction jobs
wait $PID0 $PID1 $PID2 $PID3
EXTRACT_STATUS=$?

echo ""
echo "Extraction phase completed with status: $EXTRACT_STATUS"
echo ""
echo "=== Hidden States Summary ==="
ls -lh ./data/cache/gpt2-medium/hidden_states/
du -sh ./data/cache/gpt2-medium/hidden_states/
echo ""

if [ $EXTRACT_STATUS -ne 0 ]; then
    echo "ERROR: Extraction failed! Exiting."
    exit $EXTRACT_STATUS
fi

# ==========================================
# PHASE 2: Parallel SAE Training
# ==========================================
echo "================================================================"
echo "PHASE 2: TRAINING SAEs (24 layers, 4 GPUs)"
echo "================================================================"
echo "Training configuration:"
echo "  Batch size: ${TRAIN_BATCH_SIZE}"
echo "  D_model: 1024, D_hidden: 5120"
echo "  Epochs: 20"
echo ""

# GPU 0: Train layers 0-5
CUDA_VISIBLE_DEVICES=0 python scripts/extract_and_train.py \
    --train-only \
    --layers 0,1,2,3,4,5 \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-medium \
    --save-dir ./models/checkpoints/gpt2-medium &
PID0=$!

# GPU 1: Train layers 6-11
CUDA_VISIBLE_DEVICES=1 python scripts/extract_and_train.py \
    --train-only \
    --layers 6,7,8,9,10,11 \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-medium \
    --save-dir ./models/checkpoints/gpt2-medium &
PID1=$!

# GPU 2: Train layers 12-17
CUDA_VISIBLE_DEVICES=2 python scripts/extract_and_train.py \
    --train-only \
    --layers 12,13,14,15,16,17 \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-medium \
    --save-dir ./models/checkpoints/gpt2-medium &
PID2=$!

# GPU 3: Train layers 18-23
CUDA_VISIBLE_DEVICES=3 python scripts/extract_and_train.py \
    --train-only \
    --layers 18,19,20,21,22,23 \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-medium \
    --save-dir ./models/checkpoints/gpt2-medium &
PID3=$!

echo "Training PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Waiting for all training jobs to complete..."

wait $PID0 $PID1 $PID2 $PID3
TRAIN_STATUS=$?

echo ""
echo "Training phase completed with status: $TRAIN_STATUS"
echo ""
echo "=== Trained Checkpoints Summary ==="
ls -lh ./models/checkpoints/gpt2-medium/
du -sh ./models/checkpoints/gpt2-medium/
echo ""

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "WARNING: Training had errors. Check logs."
fi

# ==========================================
# PHASE 3: Hypothesis Tracking
# ==========================================
echo "================================================================"
echo "PHASE 3: HYPOTHESIS TRACKING"
echo "================================================================"
echo ""

# Run tracking on GPU 0 (single GPU sufficient)
CUDA_VISIBLE_DEVICES=0 python tracking/hypothesis_tracker.py \
    --config ${MODEL_CONFIG} \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir ./results/gpt2-medium/tracking \
    --device cuda

TRACK_STATUS=$?

echo ""
echo "Tracking phase completed with status: $TRACK_STATUS"
echo ""

# ==========================================
# PHASE 4: Hallucination Detection
# ==========================================
echo "================================================================"
echo "PHASE 4: HALLUCINATION DETECTION"
echo "================================================================"
echo ""

# Run detection on GPU 0
CUDA_VISIBLE_DEVICES=0 python detection/detector.py \
    --config ${MODEL_CONFIG} \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir ./results/gpt2-medium/tracking \
    --output-dir ./results/gpt2-medium/detection \
    --device cuda

DETECT_STATUS=$?

echo ""
echo "Detection phase completed with status: $DETECT_STATUS"
echo ""

# ==========================================
# PHASE 5: Evaluation & Metrics
# ==========================================
echo "================================================================"
echo "PHASE 5: EVALUATION & METRICS"
echo "================================================================"
echo ""

python evaluation/pipeline.py \
    --config ${MODEL_CONFIG} \
    --results-dir ./results/gpt2-medium \
    --output-dir ./results/gpt2-medium/evaluation

EVAL_STATUS=$?

echo ""
echo "Evaluation phase completed with status: $EVAL_STATUS"
echo ""

# ==========================================
# FINAL SUMMARY
# ==========================================
echo "================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================"
echo "Job completed at $(date)"
echo ""
echo "Status Summary:"
echo "  Phase 1 (Extraction):  $EXTRACT_STATUS"
echo "  Phase 2 (Training):    $TRAIN_STATUS"
echo "  Phase 3 (Tracking):    $TRACK_STATUS"
echo "  Phase 4 (Detection):   $DETECT_STATUS"
echo "  Phase 5 (Evaluation):  $EVAL_STATUS"
echo ""
echo "Results saved to: ./results/gpt2-medium/"
echo "Checkpoints saved to: ./models/checkpoints/gpt2-medium/"
echo "================================================================"

# Calculate total time
echo ""
echo "Total job duration: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"

# Final disk usage summary
echo ""
echo "=== Disk Usage Summary ==="
du -sh ./data/cache/gpt2-medium/
du -sh ./models/checkpoints/gpt2-medium/
du -sh ./results/gpt2-medium/
