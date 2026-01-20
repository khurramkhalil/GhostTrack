#!/bin/bash
# GhostTrack Model Analysis Launcher
# Usage: ./launch_model_analysis.sh [gpt2-small|gpt2-medium|both]

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 [gpt2-small|gpt2-medium|both]"
    echo ""
    echo "Options:"
    echo "  gpt2-small   - Run analysis for GPT-2 Small (12 layers, ~15 hours)"
    echo "  gpt2-medium  - Run analysis for GPT-2 Medium (24 layers, ~60 hours)"
    echo "  both         - Run both analyses sequentially"
    exit 1
fi

MODEL=$1

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}GhostTrack Model Analysis Launcher${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

launch_gpt2_small() {
    echo -e "${GREEN}Launching GPT-2 Small Analysis...${NC}"
    echo "  Model: GPT-2 Small (117M params)"
    echo "  Layers: 12"
    echo "  Extraction batch: 32"
    echo "  Training batch: 256"
    echo "  Estimated time: ~15 hours"
    echo ""
    
    JOBID=$(sbatch jobs/full_pipeline.sbatch | awk '{print $4}')
    
    echo -e "${GREEN}✓ Job submitted: ${JOBID}${NC}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j ${JOBID}"
    echo "  tail -f gpt2s-full_${JOBID}.out"
    echo ""
    echo "Results will be in:"
    echo "  ./results/gpt2-small/"
    echo "  ./models/checkpoints/gpt2-small/"
    echo ""
}

launch_gpt2_medium() {
    echo -e "${GREEN}Launching GPT-2 Medium Analysis...${NC}"
    echo "  Model: GPT-2 Medium (345M params)"
    echo "  Layers: 24"
    echo "  Extraction batch: 16"
    echo "  Training batch: 128"
    echo "  Estimated time: ~60 hours"
    echo ""
    
    JOBID=$(sbatch jobs/gpt2_medium_full_pipeline.sbatch | awk '{print $4}')
    
    echo -e "${GREEN}✓ Job submitted: ${JOBID}${NC}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j ${JOBID}"
    echo "  tail -f gpt2m-full_${JOBID}.out"
    echo ""
    echo "Results will be in:"
    echo "  ./results/gpt2-medium/"
    echo "  ./models/checkpoints/gpt2-medium/"
    echo ""
}

case "$MODEL" in
    gpt2-small)
        launch_gpt2_small
        ;;
    gpt2-medium)
        launch_gpt2_medium
        ;;
    both)
        echo -e "${YELLOW}Launching both models...${NC}"
        echo ""
        launch_gpt2_small
        sleep 2
        launch_gpt2_medium
        echo -e "${BLUE}================================================================${NC}"
        echo -e "${GREEN}Both jobs submitted successfully!${NC}"
        echo -e "${BLUE}================================================================${NC}"
        ;;
    *)
        echo -e "${YELLOW}Error: Unknown model '${MODEL}'${NC}"
        echo "Valid options: gpt2-small, gpt2-medium, both"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}================================================================${NC}"
echo "For more information, see: docs/MODEL_CONFIGS.md"
echo -e "${BLUE}================================================================${NC}"
