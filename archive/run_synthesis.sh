#!/bin/bash
#SBATCH --job-name=ddpm_evolution
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00         
#SBATCH --output=synth_evolution.log

module purge

# Call your environment's Python path directly to execute on the GPU compute node
# /home/u6dm/mk25bm.u6dm/miniforge3/envs/ild_ddpm/bin/python -u /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/generate_synthetic_data.py
# /home/u6dm/mk25bm.u6dm/miniforge3/envs/ild_ddpm/bin/python -u /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/compile_ddpm_master_matrix.py
/home/u6dm/mk25bm.u6dm/miniforge3/envs/ild_ddpm/bin/python -u /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/sample.py
# Call your diagnostic script directly
#/home/u6dm/mk25bm.u6dm/miniforge3/envs/ild_ddpm/bin/python -u /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/debug_sampling_tensors.py 
/home/u6dm/mk25bm.u6dm/miniforge3/envs/ild_ddpm/bin/python -u /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/audit_distribution_fidelity.py