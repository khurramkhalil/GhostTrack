#!/bin/bash
#SBATCH --job-name=ghosttrack-full
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
# GhostTrack Full Pipeline - 4x A100 Parallel
# ==========================================
# Runs extraction on 4 GPUs in parallel (3 layers per GPU)
# Then trains SAEs for all 12 layers

echo "================================================================"
echo "GhostTrack Full Pipeline - 4 GPU Parallel"
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
    --num-tokens 10000000 \
    --batch-size-extract 32 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID0=$!

# GPU 1: Layers 3,4,5
CUDA_VISIBLE_DEVICES=1 python scripts/extract_and_train.py \
    --extract-only \
    --layers 3,4,5 \
    --num-tokens 10000000 \
    --batch-size-extract 32 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID1=$!

# GPU 2: Layers 6,7,8
CUDA_VISIBLE_DEVICES=2 python scripts/extract_and_train.py \
    --extract-only \
    --layers 6,7,8 \
    --num-tokens 10000000 \
    --batch-size-extract 32 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID2=$!

# GPU 3: Layers 9,10,11
CUDA_VISIBLE_DEVICES=3 python scripts/extract_and_train.py \
    --extract-only \
    --layers 9,10,11 \
    --num-tokens 10000000 \
    --batch-size-extract 32 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID3=$!

echo "Extraction PIDs: $PID0 $PID1 $PID2 $PID3"
echo "Waiting for all extraction jobs to complete..."

# Wait for all extraction jobs
wait $PID0 $PID1 $PID2 $PID3
EXTRACT_STATUS=$?

echo ""
echo "Extraction phase completed with status: $EXTRACT_STATUS"
echo "Hidden states:"
ls -la ./data/cache/hidden_states/

# ==========================================
# PHASE 2: Parallel SAE Training
# ==========================================
echo ""
echo "Starting parallel SAE training on 4 GPUs..."

# GPU 0: Train layers 0,1,2
CUDA_VISIBLE_DEVICES=0 python scripts/extract_and_train.py \
    --train-only \
    --layers 0,1,2 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID0=$!

# GPU 1: Train layers 3,4,5
CUDA_VISIBLE_DEVICES=1 python scripts/extract_and_train.py \
    --train-only \
    --layers 3,4,5 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID1=$!

# GPU 2: Train layers 6,7,8
CUDA_VISIBLE_DEVICES=2 python scripts/extract_and_train.py \
    --train-only \
    --layers 6,7,8 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID2=$!

# GPU 3: Train layers 9,10,11
CUDA_VISIBLE_DEVICES=3 python scripts/extract_and_train.py \
    --train-only \
    --layers 9,10,11 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
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
ls -la ./models/checkpoints/

echo ""
echo "=== Cache Contents ==="
du -sh ./data/cache/hidden_states/
