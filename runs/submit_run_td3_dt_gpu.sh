#!/bin/bash
#SBATCH -c 20                # Request 20 CPU cores
#SBATCH --gres=gpu:volta:1   # Request 1 Volta GPU
#SBATCH -o run_td3_dt_gpu.log-%j  # Output file
#SBATCH -J run_td3_dt_gpu_job # Job name

# Initialize conda for use in the script
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate finrl-dt

# Run your main script
bash runs/run_td3_dt_gpu.sh
