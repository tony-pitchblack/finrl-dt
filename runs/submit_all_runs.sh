#!/bin/bash

# Submit all run scripts
sbatch submit_run_a2c_dt_gpu.sh
sbatch submit_run_ddpg_dt_gpu.sh
sbatch submit_run_td3_dt_gpu.sh
sbatch submit_run_ppo_dt_gpu.sh
sbatch submit_run_sac_dt_gpu.sh
