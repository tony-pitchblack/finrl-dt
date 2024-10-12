#!/bin/bash

# Submit all run scripts
sbatch runs/run_a2c_dt_gpu.sh
sbatch runs/run_ddpg_dt_gpu.sh
sbatch runs/run_td3_dt_gpu.sh
sbatch runs/run_ppo_dt_gpu.sh
sbatch runs/run_sac_dt_gpu.sh
sbatch runs/run_ensemble_dt_gpu.sh
