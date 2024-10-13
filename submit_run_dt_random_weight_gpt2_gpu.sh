#!/bin/bash
#SBATCH -c 20                # Request 20 CPU cores
#SBATCH --gres=gpu:volta:1   # Request 1 Volta GPU
#SBATCH -o run_ddpg_dt_gpu.log-%j  # Output file
#SBATCH -J run_ddpg_dt_gpu_job # Job name

# Dynamically find the path to conda.sh based on the conda executable
CONDA_PATH=$(dirname "$(dirname "$(which conda)")")/etc/profile.d/conda.sh

# Source the conda.sh script
source "$CONDA_PATH"

# Initialize conda for use in the script
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate finrl-dt

# Run your main script
bash run_dt_random_weight_gpt2_gpu.sh