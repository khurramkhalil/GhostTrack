#!/bin/bash
#SBATCH --job-name=phi-tuning
#SBATCH --output=logs/phi_tuning_%j.out
#SBATCH --error=logs/phi_tuning_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:3
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=hoquek-lab
#SBATCH --partition=hoquek-lab-gpu

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source ~/venvs/ghosttrack/bin/activate

# Change to project directory
cd /cluster/VAST/hoquek-lab/GhostTrack

# Create directories
mkdir -p logs/phi-2
mkdir -p data/cache/phi-2
mkdir -p models/checkpoints/phi-2

echo "Starting Phi-2 Hypothesis Tracking Pipeline..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Run full pipeline with model override
# We reuse the main pipeline but point to the phi config
python3 scripts/run_detection_pipeline.py \
    --config config/model_configs/phi-2.yaml \
    --phases 1 2 3 4 \
    --parallel

echo "Pipeline finished at $(date)"
