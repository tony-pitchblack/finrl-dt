#!/bin/bash
# Run script for IQL with multiple algorithms

# Activate the appropriate conda environment if needed
# conda activate your_environment_name

# Set environment variables if needed
# export SOME_VAR=some_value

# Define the seeds
seeds=(20742 55230 85125 96921 67851)

# Define the algorithms
algos=("a2c" "ppo" "ddpg" "sac" "td3")
# algos=("ppo" "ddpg" "sac" "td3")

# Common parameters
gpu=-1 # GPU identification number

# IQL specific parameters (adjust these as needed for your iql.py script)
# Add any other parameters your iql.py script expects

for algo in "${algos[@]}"; do
    # Find the matching dataset for the current algorithm
    dataset_path=$(find data/ -name "train_${algo}_trajectory_*" | sort -r | head -n 1)
    test_trajectory=$(find data/ -name "test_${algo}_trajectory_*" | sort -r | head -n 1)
    
    if [ -z "$dataset_path" ]; then
        echo "No dataset found for algorithm: $algo. Skipping..."
        continue
    fi

    for seed in "${seeds[@]}"; do
        echo "Running IQL with algorithm: $algo and seed: $seed"
        echo "Using dataset: $dataset_path"
        
        CUDA_VISIBLE_DEVICES=$gpu ~/gits/FinRL-Tutorials/.conda/bin/python cql.py \
            --seed $seed \
            --drl_algo $algo \
            --dataset_path "$dataset_path" \
            --test_trajectory "$test_trajectory"
            # Add any other command-line arguments your iql.py script expects
        
        echo "Finished run with algorithm: $algo and seed: $seed"
        echo "------------------------"
    done
done

echo "All runs completed."
