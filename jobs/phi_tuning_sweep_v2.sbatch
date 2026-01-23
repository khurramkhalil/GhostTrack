#!/bin/bash
#SBATCH --job-name=phi-sweep-v2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=hoquek-lab
#SBATCH --partition=gpu
#SBATCH --exclude=g008,g039
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Load modules
module load miniconda3/4.10.3_gcc_9.5.0
module load cuda/11.8

# Activate environment
source activate deepseek

# Change to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

# Result directories
RESULTS_DIR="./results/phi-2/tuning_v2"
MODEL_DIR="./models/checkpoints/phi-2"

mkdir -p $RESULTS_DIR

# Base Config (Phi-2 default)
BASE_CONFIG="config/model_configs/phi-2.yaml"

# --- Experiment 1: Ultra Sensitivity ---
# Top-k=200, Thresh=0.2, Sem=0.4
echo "--- Running Experiment 1: Ultra Sensitivity ---"
EXP1_CONFIG="config/model_configs/phi-2-ultra.yaml"
cp $BASE_CONFIG $EXP1_CONFIG
sed -i 's/top_k_features: 50/top_k_features: 200/' $EXP1_CONFIG
sed -i 's/birth_threshold: 0.5/birth_threshold: 0.2/' $EXP1_CONFIG
sed -i 's/semantic_weight: 0.6/semantic_weight: 0.4/' $EXP1_CONFIG

python3 scripts/run_tracking.py \
    --config $EXP1_CONFIG \
    --model-dir $MODEL_DIR \
    --output-dir $RESULTS_DIR/ultra/tracking \
    --device cuda

python3 scripts/run_detection.py \
    --config $EXP1_CONFIG \
    --model-dir $MODEL_DIR \
    --tracking-dir $RESULTS_DIR/ultra/tracking \
    --output-dir $RESULTS_DIR/ultra/detection \
    --device cuda


# --- Experiment 2: High Semantic Weight ---
# Top-k=100, Thresh=0.3, Sem=0.8
echo "--- Running Experiment 2: High Semantic Weight ---"
EXP2_CONFIG="config/model_configs/phi-2-highsem.yaml"
cp $BASE_CONFIG $EXP2_CONFIG
sed -i 's/top_k_features: 50/top_k_features: 100/' $EXP2_CONFIG
sed -i 's/birth_threshold: 0.5/birth_threshold: 0.3/' $EXP2_CONFIG
sed -i 's/semantic_weight: 0.6/semantic_weight: 0.8/' $EXP2_CONFIG

python3 scripts/run_tracking.py \
    --config $EXP2_CONFIG \
    --model-dir $MODEL_DIR \
    --output-dir $RESULTS_DIR/high-sem/tracking \
    --device cuda

python3 scripts/run_detection.py \
    --config $EXP2_CONFIG \
    --model-dir $MODEL_DIR \
    --tracking-dir $RESULTS_DIR/high-sem/tracking \
    --output-dir $RESULTS_DIR/high-sem/detection \
    --device cuda


# --- Experiment 3: Low Semantic Weight ---
# Top-k=100, Thresh=0.3, Sem=0.1
echo "--- Running Experiment 3: Low Semantic Weight ---"
EXP3_CONFIG="config/model_configs/phi-2-lowsem.yaml"
cp $BASE_CONFIG $EXP3_CONFIG
sed -i 's/top_k_features: 50/top_k_features: 100/' $EXP3_CONFIG
sed -i 's/birth_threshold: 0.5/birth_threshold: 0.3/' $EXP3_CONFIG
sed -i 's/semantic_weight: 0.6/semantic_weight: 0.1/' $EXP3_CONFIG

python3 scripts/run_tracking.py \
    --config $EXP3_CONFIG \
    --model-dir $MODEL_DIR \
    --output-dir $RESULTS_DIR/low-sem/tracking \
    --device cuda

python3 scripts/run_detection.py \
    --config $EXP3_CONFIG \
    --model-dir $MODEL_DIR \
    --tracking-dir $RESULTS_DIR/low-sem/tracking \
    --output-dir $RESULTS_DIR/low-sem/detection \
    --device cuda


# --- Experiment 4: Loose Association ---
# Top-k=100, Thresh=0.3, Assoc=0.3
echo "--- Running Experiment 4: Loose Association ---"
EXP4_CONFIG="config/model_configs/phi-2-loose.yaml"
cp $BASE_CONFIG $EXP4_CONFIG
sed -i 's/top_k_features: 50/top_k_features: 100/' $EXP4_CONFIG
sed -i 's/birth_threshold: 0.5/birth_threshold: 0.3/' $EXP4_CONFIG
sed -i 's/association_threshold: 0.5/association_threshold: 0.3/' $EXP4_CONFIG

python3 scripts/run_tracking.py \
    --config $EXP4_CONFIG \
    --model-dir $MODEL_DIR \
    --output-dir $RESULTS_DIR/loose/tracking \
    --device cuda

python3 scripts/run_detection.py \
    --config $EXP4_CONFIG \
    --model-dir $MODEL_DIR \
    --tracking-dir $RESULTS_DIR/loose/tracking \
    --output-dir $RESULTS_DIR/loose/detection \
    --device cuda

echo "Tuning Sweep V2 Complete"
