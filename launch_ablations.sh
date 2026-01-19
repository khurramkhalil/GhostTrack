#!/bin/bash

# Configuration
CLUSTER_ALIAS="hell"
REMOTE_DIR="/cluster/VAST/hoquek-lab/GhostTrack"
LOCAL_DIR="/Users/khurram/Documents/GhostTrack"

echo "================================================================"
echo "Launching GhostTrack Ablation Studies (Phase 6)"
echo "Using SSH alias: $CLUSTER_ALIAS"
echo "================================================================"

# 1. Upload Scripts
echo "[1/2] Uploading scripts..."
# Note: Using 'hell' alias which should act as user@host
scp -r $LOCAL_DIR/scripts/run_ablations.py $CLUSTER_ALIAS:$REMOTE_DIR/scripts/
scp -r $LOCAL_DIR/scripts/run_detection_pipeline.py $CLUSTER_ALIAS:$REMOTE_DIR/scripts/
scp -r $LOCAL_DIR/evaluation/pipeline.py $CLUSTER_ALIAS:$REMOTE_DIR/evaluation/
scp -r $LOCAL_DIR/tracking/track_association.py $CLUSTER_ALIAS:$REMOTE_DIR/tracking/
scp -r $LOCAL_DIR/tracking/hypothesis_tracker.py $CLUSTER_ALIAS:$REMOTE_DIR/tracking/
scp -r $LOCAL_DIR/tracking/feature_extractor.py $CLUSTER_ALIAS:$REMOTE_DIR/tracking/
scp -r $LOCAL_DIR/models/model_wrapper.py $CLUSTER_ALIAS:$REMOTE_DIR/models/
scp -r $LOCAL_DIR/detection/detector.py $CLUSTER_ALIAS:$REMOTE_DIR/detection/
scp -r $LOCAL_DIR/detection/divergence_metrics.py $CLUSTER_ALIAS:$REMOTE_DIR/detection/
scp -r $LOCAL_DIR/jobs/ablations.sbatch $CLUSTER_ALIAS:$REMOTE_DIR/jobs/

if [ $? -ne 0 ]; then
    echo "Upload failed!"
    exit 1
fi

# 2. Submit Job
echo "[2/2] Submitting job..."
ssh $CLUSTER_ALIAS "cd $REMOTE_DIR && sbatch jobs/ablations.sbatch"

if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Use 'squeue -u khurram' to check status."
else
    echo "Job submission failed!"
fi
