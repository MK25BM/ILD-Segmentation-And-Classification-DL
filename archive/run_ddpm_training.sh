#!/bin/bash
#SBATCH --job-name=custom_ild_ddpm
#SBATCH --partition=workq       # FIX: Explicitly targets the workq hardware channel lane
#SBATCH --nodes=1
#SBATCH --gres=gpu:1            # Natively claims 1 complete GH200 120GB Superchip!
#SBATCH --cpus-per-task=8       # Allocates 8 CPU cores to drive your multi-threaded DataLoader workers
#SBATCH --mem=64G
#SBATCH --time=02:30:00        
#SBATCH --output=custom_ddpm_%j.log

# Clean hardware system re-alignment
module purge

# Enforce clean single-threaded CPU scaling to completely kill OpenBLAS scheduler lag
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PYTORCH_ALLOC_CONF="expandable_segments:True"

/home/u6dm/mk25bm.u6dm/miniforge3/envs/ild_ddpm/bin/python -u /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/run_ddpm_training.py
