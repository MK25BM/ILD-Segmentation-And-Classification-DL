#!/bin/bash
#SBATCH --job-name=medgift_manifest
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=manifest_pipeline_%j.log

# 1. Load the cluster's native CUDA compiler module (Crucial for Grace Hopper)
module load nvhpc/24.11


# 2. Activate your newly built environment
source ~/miniforge3/bin/activate
conda activate ild_ddpm

# 3. Execute the updated programmatic manifest script
#python ~/ILD-Segmentation-And-Classification-DL/generate_manifest.py

# 4. Execute the 2d slice extraction script
# Double check the bottom line of run_preprocess.sh points to your script:
python ~/ILD-Segmentation-And-Classification-DL/extract_2d_slices.py