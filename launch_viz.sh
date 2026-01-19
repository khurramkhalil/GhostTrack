#!/bin/bash

# Configuration
CLUSTER_HOST="hell"
CLUSTER_DIR="/cluster/VAST/hoquek-lab/GhostTrack"

echo "================================================================"
echo "GhostTrack Visualization Launcher"
echo "================================================================"

# 1. Upload scripts
echo "[1/2] Uploading scripts to cluster..."
scp scripts/run_visualization.py $CLUSTER_HOST:$CLUSTER_DIR/scripts/
scp jobs/visualization_pipeline.sbatch $CLUSTER_HOST:$CLUSTER_DIR/jobs/
echo "✓ Scripts uploaded"

# 2. Submit Job
echo ""
echo "[2/2] Submitting Visualization Job..."
ssh $CLUSTER_HOST "cd $CLUSTER_DIR && sbatch jobs/visualization_pipeline.sbatch"

echo ""
echo "================================================================"
echo "Job Submitted! Monitor with:"
echo "ssh $CLUSTER_HOST squeue -u \$(whoami)"
echo "================================================================"
