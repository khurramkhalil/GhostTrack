#!/bin/bash
# Manual deployment steps for Hellbender
# Run this ON Hellbender after SSHing into it

cd /cluster/VAST/hoquek-lab/GhostTrack

# Pull latest changes
echo "Pulling latest changes from GitHub..."
git pull

# Make scripts executable
chmod +x launch_model_analysis.sh

# Verify the setup
echo ""
echo "Verifying files..."
ls -la jobs/gpt2_medium_full_pipeline.sbatch
ls -la config/model_configs/gpt2-medium.yaml
ls -la launch_model_analysis.sh

# Launch the GPT-2 Medium pipeline
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Launching GPT-2 Medium Pipeline..."
echo "════════════════════════════════════════════════════════════"
echo ""

./launch_model_analysis.sh gpt2-medium

echo ""
echo "Job submitted! To monitor:"
echo "  squeue -u \$USER"
echo "  tail -f gpt2m-full_*.out"
