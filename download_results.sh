#!/bin/bash

# Configuration
CLUSTER_HOST="hell"  # As used in your ssh command
CLUSTER_DIR="/cluster/VAST/hoquek-lab/GhostTrack"
LOCAL_DIR="./results_downloaded"

echo "================================================================"
echo "GhostTrack Smart Downloader"
echo "================================================================"
echo "Downloading results from $CLUSTER_HOST..."
echo "Target: $LOCAL_DIR"

# Create local directory
mkdir -p $LOCAL_DIR

# 1. Download Detection Results (Tiny - Priority)
echo ""
echo "[1/3] Downloading Detection Results (< 10 MB)..."
scp -r $CLUSTER_HOST:$CLUSTER_DIR/results/detection $LOCAL_DIR/
echo "✓ Detection results downloaded"

# 2. Download Training Logs (Tiny)
echo ""
echo "[2/3] Downloading Logs (< 1 MB)..."
scp $CLUSTER_HOST:$CLUSTER_DIR/*.out $LOCAL_DIR/ 2>/dev/null
scp $CLUSTER_HOST:$CLUSTER_DIR/*.err $LOCAL_DIR/ 2>/dev/null
echo "✓ Logs downloaded"

# 3. Download Visualization Results
echo ""
echo "[3/4] Downloading Visualization Results..."
scp -r $CLUSTER_HOST:$CLUSTER_DIR/results/visualization $LOCAL_DIR/
echo "✓ Visualization results downloaded"

# 3. Optional: SAE Checkpoints (Medium - ~300 MB)
# Uncomment the line below if you want the trained SAE models
# echo ""
# echo "[3/3] Downloading Trained SAEs (~300 MB)..."
# mkdir -p $LOCAL_DIR/models
# scp -r $CLUSTER_HOST:$CLUSTER_DIR/models/checkpoints $LOCAL_DIR/models/

echo ""
echo "================================================================"
echo "Download Complete!"
echo "Files are in: $LOCAL_DIR"
echo "NOTE: Large 'data/cache' folder (>300GB) was skipped."
echo "================================================================"
