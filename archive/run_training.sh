#!/bin/bash
#SBATCH --job-name=ild_segmentation_experiment
#SBATCH --partition=workq
#SBATCH --nodes=1                      # Requests exactly 1 compute node
#SBATCH --gres=gpu:1                   # FIXED: Use standard gres mapping to secure the GH200 device path
#SBATCH --cpus-per-task=8              # Allocates 8 clean CPU cores for background workers
#SBATCH --mem=64G                       # Safe, massive 64GB system RAM boundary
#SBATCH --time=01:00:00                 # OPTIMIZED: 3.5 hours instantly triggers backfill queues!
#SBATCH --output=unet_experiment_%j.log

# Clean hardware system re-alignment
module purge

# Enforce clean single-threaded CPU scaling to completely kill OpenBLAS scheduler lag
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Instructs PyTorch to append memory blocks sequentially, preventing fragmentation OOM errors
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ─── 🚀 THE DIRECT BINARY BYPASS ENGINE ───
# Calls your environment's Python binary directly via its absolute path.
# This completely bypasses 'conda activate' path bugs and guarantees the GPU is claimed!
/home/u6dm/mk25bm.u6dm/miniforge3/envs/ild_ddpm/bin/python -u /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/run_leakage_proof_experiment.py --model attention_unet --epochs 60
