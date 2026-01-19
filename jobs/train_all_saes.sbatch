#!/bin/bash
#SBATCH --job-name=ghosttrack-train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=24:00:00

# ==========================================
# GhostTrack Phase 2 ONLY - Optimized Training
# ==========================================
# Skips extraction, trains 12 SAEs on existing hidden states
# Uses optimized batch size and parallel data loading

echo "================================================================"
echo "GhostTrack Phase 2 - Optimized SAE Training"
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

# Verify hidden states exist
echo ""
echo "=== Verifying Phase 1 data ==="
ls -la ./data/cache/hidden_states/ | head -5
echo "..."
LAYER_COUNT=$(ls ./data/cache/hidden_states/layer_*_states.pt 2>/dev/null | wc -l)
echo "Found $LAYER_COUNT layer files"

if [ "$LAYER_COUNT" -ne 12 ]; then
    echo "ERROR: Expected 12 layer files, found $LAYER_COUNT"
    exit 1
fi

echo ""
echo "=== Starting Parallel SAE Training (4 GPUs × 3 layers) ==="

# GPU 0: Train layers 0,1,2
CUDA_VISIBLE_DEVICES=0 python scripts/extract_and_train.py \
    --train-only \
    --layers 0,1,2 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID0=$!
echo "GPU 0 (layers 0,1,2): PID $PID0"

# GPU 1: Train layers 3,4,5
CUDA_VISIBLE_DEVICES=1 python scripts/extract_and_train.py \
    --train-only \
    --layers 3,4,5 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID1=$!
echo "GPU 1 (layers 3,4,5): PID $PID1"

# GPU 2: Train layers 6,7,8
CUDA_VISIBLE_DEVICES=2 python scripts/extract_and_train.py \
    --train-only \
    --layers 6,7,8 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID2=$!
echo "GPU 2 (layers 6,7,8): PID $PID2"

# GPU 3: Train layers 9,10,11
CUDA_VISIBLE_DEVICES=3 python scripts/extract_and_train.py \
    --train-only \
    --layers 9,10,11 \
    --device cuda \
    --cache-dir ./data/cache \
    --save-dir ./models/checkpoints &
PID3=$!
echo "GPU 3 (layers 9,10,11): PID $PID3"

echo ""
echo "Waiting for all training jobs to complete..."
wait $PID0 $PID1 $PID2 $PID3
TRAIN_STATUS=$?

echo ""
echo "================================================================"
echo "Training completed at $(date)"
echo "Exit status: $TRAIN_STATUS"
echo "================================================================"

echo ""
echo "=== Trained Checkpoints ==="
ls -la ./models/checkpoints/

echo ""
echo "=== Summary ==="
BEST_COUNT=$(ls ./models/checkpoints/sae_layer_*_best.pt 2>/dev/null | wc -l)
FINAL_COUNT=$(ls ./models/checkpoints/sae_layer_*_final.pt 2>/dev/null | wc -l)
echo "Best checkpoints: $BEST_COUNT/12"
echo "Final checkpoints: $FINAL_COUNT/12"
