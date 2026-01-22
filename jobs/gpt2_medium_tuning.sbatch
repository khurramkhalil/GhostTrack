#!/bin/bash
#SBATCH --job-name=gpt2m-tune
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# ==========================================
# GhostTrack Hyperparameter Tuning - GPT-2 Medium
# ==========================================
# Runs 4 experimental configurations to find optimal tracking parameters

echo "================================================================"
echo "GhostTrack GPT-2 Medium - Hyperparameter Tuning"
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# Load environment
module load miniconda3/4.10.3_gcc_9.5.0
source activate deepseek
cd /cluster/VAST/hoquek-lab/GhostTrack

# Check GPU
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Base config
BASE_CONFIG="./config/model_configs/gpt2-medium.yaml"
RESULTS_BASE="./results/gpt2-medium/tuning"
NUM_SAMPLES=150  # Enough to get signal, fast enough for tuning

mkdir -p $RESULTS_BASE
mkdir -p ./config/tuning

# Helper to generate config
create_config() {
    local name=$1
    local top_k=$2
    local threshold=$3
    local semantic=$4
    local outfile="./config/tuning/${name}.yaml"
    
    echo "Creating config for $name: top_k=$top_k, thresh=$threshold, sem=$semantic"
    
    cp $BASE_CONFIG $outfile
    
    # Use sed to replace parameters (mac/linux compatible sed)
    sed -i "s/top_k_features: .*/top_k_features: $top_k/" $outfile
    sed -i "s/association_threshold: .*/association_threshold: $threshold/" $outfile
    sed -i "s/semantic_weight: .*/semantic_weight: $semantic/" $outfile
    
    # Also fix paths to avoid overwriting main results
    sed -i "s|results_dir: .*|results_dir: $RESULTS_BASE/$name|" $outfile
}

# ==========================================
# 1. BASELINE (top_k=50, assoc=0.5)
# ==========================================
echo ""
echo "--- Running Experiment 1: Baseline ---"
create_config "baseline" 50 0.5 0.6

python scripts/run_tracking.py \
    --config ./config/tuning/baseline.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir $RESULTS_BASE/baseline/tracking \
    --num-samples $NUM_SAMPLES

python scripts/run_detection.py \
    --config ./config/tuning/baseline.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir $RESULTS_BASE/baseline/tracking \
    --output-dir $RESULTS_BASE/baseline/detection

# ==========================================
# 2. HIGH RECALL (top_k=100, assoc=0.45)
# ==========================================
echo ""
echo "--- Running Experiment 2: High Recall ---"
create_config "high_recall" 100 0.45 0.6

python scripts/run_tracking.py \
    --config ./config/tuning/high_recall.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir $RESULTS_BASE/high_recall/tracking \
    --num-samples $NUM_SAMPLES

python scripts/run_detection.py \
    --config ./config/tuning/high_recall.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir $RESULTS_BASE/high_recall/tracking \
    --output-dir $RESULTS_BASE/high_recall/detection

# ==========================================
# 3. HIGH PRECISION (top_k=30, assoc=0.6)
# ==========================================
echo ""
echo "--- Running Experiment 3: High Precision ---"
create_config "high_precision" 30 0.6 0.6

python scripts/run_tracking.py \
    --config ./config/tuning/high_precision.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir $RESULTS_BASE/high_precision/tracking \
    --num-samples $NUM_SAMPLES

python scripts/run_detection.py \
    --config ./config/tuning/high_precision.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir $RESULTS_BASE/high_precision/tracking \
    --output-dir $RESULTS_BASE/high_precision/detection

# ==========================================
# 4. RELAXED ALIGNMENT (top_k=50, assoc=0.75, sem=0.8)
# ==========================================
echo ""
echo "--- Running Experiment 4: Relaxed Alignment ---"
# Sim=0.2 -> Cost=0.8*0.8 = 0.64 < 0.75 (Match)
# Sim=0.0 -> Cost=0.8*1.0 = 0.80 > 0.75 (No Match)
create_config "relaxed_alignment" 50 0.75 0.8

python scripts/run_tracking.py \
    --config ./config/tuning/relaxed_alignment.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --output-dir $RESULTS_BASE/relaxed_alignment/tracking \
    --num-samples $NUM_SAMPLES

python scripts/run_detection.py \
    --config ./config/tuning/relaxed_alignment.yaml \
    --model-dir ./models/checkpoints/gpt2-medium \
    --tracking-dir $RESULTS_BASE/relaxed_alignment/tracking \
    --output-dir $RESULTS_BASE/relaxed_alignment/detection

# ==========================================
# SUMMARY
# ==========================================
echo ""
echo "================================================================"
echo "TUNING COMPLETE"
echo "================================================================"
echo "Results summary:"

for exp in baseline high_recall high_precision relaxed_alignment; do
    echo ""
    echo "Experiment: $exp"
    grep "AUROC" $RESULTS_BASE/$exp/detection/detection_metrics.json 2>/dev/null || echo "Metrics not found"
    grep "Accuracy" $RESULTS_BASE/$exp/detection/detection_metrics.json 2>/dev/null
done

echo ""
echo "Job completed at $(date)"
