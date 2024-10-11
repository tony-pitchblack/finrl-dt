#!/bin/bash
#SBATCH -c 20                # Request 20 CPU cores
#SBATCH --gres=gpu:volta:1   # Request 1 Volta GPU
#SBATCH -o run_stock_trading.log-%j  # Output file
#SBATCH -J stock_trading_job # Job name (optional)

# Activate your conda environment
source ~/.bashrc
conda activate finrl-dt

# Export environment variables (if needed)
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

# Run your script
bash run_a2c_dt_gpu.sh
