#!/bin/bash
# Script to promote best tuning results to main results directory
# Run on cluster
# Usage: ./scripts/promote_results.sh

SRC_DIR="./results/gpt2-medium/tuning/baseline"
DEST_DIR="./results/gpt2-medium"

echo "Promoting baseline results to main directory..."

# Backup existing if any
if [ -d "$DEST_DIR/tracking" ]; then
    echo "Backing up existing tracking results..."
    mv "$DEST_DIR/tracking" "$DEST_DIR/tracking_backup_$(date +%Y%m%d_%H%M%S)"
fi

if [ -d "$DEST_DIR/detection" ]; then
    echo "Backing up existing detection results..."
    mv "$DEST_DIR/detection" "$DEST_DIR/detection_backup_$(date +%Y%m%d_%H%M%S)"
fi

# Copy results
echo "Copying from $SRC_DIR to $DEST_DIR..."
mkdir -p "$DEST_DIR"
cp -r "$SRC_DIR/tracking" "$DEST_DIR/"
cp -r "$SRC_DIR/detection" "$DEST_DIR/"

# Run visualization (Phase 5)
echo "Running Phase 5: Visualization..."
python3 scripts/run_visualization.py \
    --model-name gpt2-medium \
    --checkpoint-dir ./models/checkpoints/gpt2-medium \
    --detector-path "$DEST_DIR/detection/detector.pkl" \
    --output-dir "$DEST_DIR/visualization"

echo "Promotion and visualization complete!"
ls -l "$DEST_DIR/visualization"
