#!/bin/bash

# Define arrays for algos and seeds
algos=("a2c" "ppo" "ddpg" "sac" "td3")
seeds=(20742 55230 85125 96921 67851)

# Loop through each algo and seed to transfer files
for algo in "${algos[@]}"; do
    for seed in "${seeds[@]}"; do
        # Define remote and local paths
        remote_dir="/home/gridsan/syun/finrl-dt/checkpoints/${algo}_dt_lora_random_weight_gpt2_${seed}/"
        local_dir="checkpoints/${algo}_bc_${seed}/"

        # Perform the rsync transfer, excluding model.pt
        rsync -avz --exclude='model.pt' "syyun@76.119.237.252:${remote_dir}" "${local_dir}"

        # Check if rsync was successful
        if [ $? -eq 0 ]; then
            echo "Successfully copied files to ${local_dir}"
        else
            echo "Failed to copy files to ${local_dir}"
        fi
    done
done

echo "All transfers completed."

