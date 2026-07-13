#!/bin/bash
#SBATCH --job-name=ild_baseline_segmentation
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=baseline_experiment_%j.log

# ─── Clean environment ───
module purge

# ─── Disable CPU threading (prevent OpenBLAS lag) ───
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ─── PyTorch memory management ───
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ─── Run training ───
/home/u6dm/mk25bm.u6dm/miniforge3/envs/ild_ddpm/bin/python -u \
    /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/scripts/train_optimized.py \
    --config training.yaml \
    --data-root /scratch/u6dm/mk25bm.u6dm/ild_dataset_processed