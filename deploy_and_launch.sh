#!/bin/bash
# Deploy changes to Hellbender and launch GPT-2 Medium pipeline

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Deploying GhostTrack to Hellbender & Launching Pipeline  ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Commit and push local changes
echo -e "${GREEN}Step 1: Committing local changes...${NC}"
git add .
git commit -m "Add GPT-2 Medium pipeline with optimized batch sizes and organized structure" || echo "Nothing to commit"
echo ""

echo -e "${GREEN}Step 2: Pushing to remote...${NC}"
git push
echo ""

# Step 2: Deploy to Hellbender and launch
echo -e "${GREEN}Step 3: Deploying to Hellbender and launching job...${NC}"
echo ""

ssh hellbender << 'ENDSSH'
cd /cluster/VAST/hoquek-lab/GhostTrack

# Pull latest changes
echo "Pulling latest changes..."
git pull

# Make launch script executable
chmod +x launch_model_analysis.sh

# Show current directory structure
echo ""
echo "Verifying setup..."
ls -la jobs/gpt2_medium_full_pipeline.sbatch
ls -la config/model_configs/gpt2-medium.yaml

# Launch the job
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Launching GPT-2 Medium Pipeline..."
echo "════════════════════════════════════════════════════════════"
echo ""

./launch_model_analysis.sh gpt2-medium

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Job submitted! Monitoring commands:"
echo "════════════════════════════════════════════════════════════"
echo "  squeue -u \$USER"
echo "  tail -f gpt2m-full_*.out"
echo ""

ENDSSH

echo ""
echo -e "${GREEN}✓ Deployment and launch complete!${NC}"
echo ""
echo -e "${YELLOW}To monitor the job:${NC}"
echo "  ssh hellbender"
echo "  cd /cluster/VAST/hoquek-lab/GhostTrack"
echo "  squeue -u \$USER"
echo "  tail -f gpt2m-full_*.out"
echo ""
