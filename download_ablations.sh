#!/bin/bash

# Download ablation results
mkdir -p results_downloaded/ablations

echo "Downloading ablation results from 'hell'..."
scp hell:/cluster/VAST/hoquek-lab/GhostTrack/results/ablations/ablation_results.json ./results_downloaded/ablations/

echo "Downloading logs..."
scp hell:'/cluster/VAST/hoquek-lab/GhostTrack/ghosttrack-ablations_*.out' ./results_downloaded/ablations/
scp hell:'/cluster/VAST/hoquek-lab/GhostTrack/ghosttrack-ablations_*.err' ./results_downloaded/ablations/

echo "Download complete. Check results_downloaded/ablations/"
