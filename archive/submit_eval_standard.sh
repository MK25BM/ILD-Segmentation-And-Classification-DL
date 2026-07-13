#!/bin/bash
#SBATCH --job-name=eval_standard_unet
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=eval_standard_unet_%j.log

# Activate your custom environment
source ~/miniforge3/bin/activate ild_ddpm

# Execute the script cleanly on the assigned GPU node
python -u /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/evaluate_holdout_test_set.py
