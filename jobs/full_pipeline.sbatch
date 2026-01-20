#!/bin/bash
#SBATCH --job-name=gpt2s-full
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
# GhostTrack Full Pipeline - GPT-2 Small
# ==========================================
# Complete end-to-end pipeline for GPT-2 Small (12 layers)
# Running on 4x A100 GPUs with optimized batch sizes

echo "================================================================"
echo "GhostTrack GPT-2 Small - Full Pipeline"
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# Load modules and activate environment
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek

# Navigate to project directory (VAST storage - 5TB available)
cd /cluster/VAST/hoquek-lab/GhostTrack

# Check GPU availability
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Configuration
MODEL_CONFIG="./config/model_configs/gpt2-small.yaml"
NUM_TOKENS=10000000
EXTRACT_BATCH_SIZE=32  # Good for A100 with GPT-2 small
TRAIN_BATCH_SIZE=256   # Optimized for A100

echo ""
echo "Configuration:"
echo "  Model: GPT-2 Small (12 layers, 768 dim)"
echo "  Tokens per layer: ${NUM_TOKENS}"
echo "  Extraction batch size: ${EXTRACT_BATCH_SIZE}"
echo "  Training batch size: ${TRAIN_BATCH_SIZE}"
echo ""
echo "Launching 4 parallel extraction jobs (3 layers each)..."

# ==========================================
# PHASE 1: Parallel Hidden State Extraction  
# ==========================================
# Run 4 processes in parallel, each handling 3 layers

# GPU 0: Layers 0,1,2
CUDA_VISIBLE_DEVICES=0 python scripts/extract_and_train.py \
    --extract-only \
    --layers 0,1,2 \
    --num-tokens ${NUM_TOKENS} \
    --batch-size-extract ${EXTRACT_BATCH_SIZE} \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-small \
    --save-dir ./models/checkpoints/gpt2-small &
PID0=$!

# GPU 1: Layers 3,4,5
CUDA_VISIBLE_DEVICES=1 python scripts/extract_and_train.py \
    --extract-only \
    --layers 3,4,5 \
    --num-tokens ${NUM_TOKENS} \
    --batch-size-extract ${EXTRACT_BATCH_SIZE} \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-small \
    --save-dir ./models/checkpoints/gpt2-small &
PID1=$!

# GPU 2: Layers 6,7,8
CUDA_VISIBLE_DEVICES=2 python scripts/extract_and_train.py \
    --extract-only \
    --layers 6,7,8 \
    --num-tokens ${NUM_TOKENS} \
    --batch-size-extract ${EXTRACT_BATCH_SIZE} \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-small \
    --save-dir ./models/checkpoints/gpt2-small &
PID2=$!

# GPU 3: Layers 9,10,11
CUDA_VISIBLE_DEVICES=3 python scripts/extract_and_train.py \
    --extract-only \
    --layers 9,10,11 \
    --num-tokens ${NUM_TOKENS} \
    --batch-size-extract ${EXTRACT_BATCH_SIZE} \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-small \
    --save-dir ./models/checkpoints/gpt2-small &
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
ls -lh ./data/cache/gpt2-small/hidden_states/
du -sh ./data/cache/gpt2-small/hidden_states/
echo ""

# ==========================================
# PHASE 2: Parallel SAE Training
# ==========================================
echo ""
echo "Starting parallel SAE training on 4 GPUs..."

# GPU 0: Train layers 0,1,2
CUDA_VISIBLE_DEVICES=0 python scripts/extract_and_train.py \
    --train-only \
    --layers 0,1,2 \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-small \
    --save-dir ./models/checkpoints/gpt2-small &
PID0=$!

# GPU 1: Train layers 3,4,5
CUDA_VISIBLE_DEVICES=1 python scripts/extract_and_train.py \
    --train-only \
    --layers 3,4,5 \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-small \
    --save-dir ./models/checkpoints/gpt2-small &
PID1=$!

# GPU 2: Train layers 6,7,8
CUDA_VISIBLE_DEVICES=2 python scripts/extract_and_train.py \
    --train-only \
    --layers 6,7,8 \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-small \
    --save-dir ./models/checkpoints/gpt2-small &
PID2=$!

# GPU 3: Train layers 9,10,11
CUDA_VISIBLE_DEVICES=3 python scripts/extract_and_train.py \
    --train-only \
    --layers 9,10,11 \
    --config ${MODEL_CONFIG} \
    --device cuda \
    --cache-dir ./data/cache/gpt2-small \
    --save-dir ./models/checkpoints/gpt2-small &
PID3=$!

echo "Training PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Waiting for all training jobs to complete..."

wait $PID0 $PID1 $PID2 $PID3
TRAIN_STATUS=$?

echo ""
echo "================================================================"
echo "Pipeline completed at $(date)"
echo "Extraction status: $EXTRACT_STATUS"
echo "Training status: $TRAIN_STATUS"
echo "================================================================"

echo ""
echo "=== Trained Checkpoints ==="
ls -lh ./models/checkpoints/gpt2-small/
du -sh ./models/checkpoints/gpt2-small/

echo ""
echo "=== Cache Contents ==="
du -sh ./data/cache/gpt2-small/hidden_states/

echo ""
echo "Total job duration: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
