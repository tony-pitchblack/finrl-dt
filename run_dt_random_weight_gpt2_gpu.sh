#!/bin/bash
# run with: source run_a2c_bc_cpu.sh

export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

gpu=0 # gpu identification number

# Training parameters
lr=1e-3
weight_decay=1e-5 # for AdamW optimizer
dropout=0.1
warmup_steps=2500
num_steps=1000 # total number of training steps; i.e., how many times we call env.step() for training

# Environment parameters
env=stock_trading
dataset=your_dataset_name
sample_ratio=0.1
K=20  # Context length - this is for the decision transformer model; Seeing the K number of states in the past to make a prediction on next action

# Device
device='cuda' # or 'cuda'

# Pretrained language model
# pretrained_lm="gpt2" # this will trigger auto-downloading the gpt2 model from the Hugging Face model hub
pretrained_lm="/home/gridsan/syun/gpt2_model" # Path to the downloaded GPT-2 model
lora=true

# Seeds for multiple runs
seeds=(20742 55230 85125 96921 67851)

algos=("a2c" "ppo" "ddpg" "sac" "td3")

for algo in "${algos[@]}"; do
    dataset_path=$(find data/ -name "train_${algo}_trajectory_*" | sort -r | head -n 1)
    test_trajectory_file=$(find data/ -name "test_${algo}_trajectory_*" | sort -r | head -n 1)
    
    if [ -z "$dataset_path" ] || [ -z "$test_trajectory_file" ]; then
        echo "Dataset or test trajectory not found for algorithm: $algo. Skipping..."
        continue
    fi

    for seed in "${seeds[@]}"; do
        exp_name="${algo}_dt_lora_random_weight_gpt2_${seed}"
        outdir="checkpoints/${exp_name}"

        echo "Running DT with algorithm: $algo and seed: $seed"
        echo "Using dataset: $dataset_path"
        echo "Using test trajectory: $test_trajectory_file"
        
        CUDA_VISIBLE_DEVICES=${gpu} python experiment.py \
            --device ${device} \
            --env stock_trading \
            --dataset_path ${dataset_path} \
            --seed ${seed} \
            --K ${K} \
            --learning_rate ${lr} \
            --num_steps ${num_steps} \
            --weight_decay ${weight_decay} \
            --sample_ratio ${sample_ratio} \
            --warmup_steps ${warmup_steps} \
            --pretrained_lm ${pretrained_lm} \
            --outdir ${outdir} \
            --dropout ${dropout} \
            --mlp_embedding \
            --adapt_mode \
            --adapt_embed \
            --lora \
            --exp_name ${exp_name} \
            --drl_alg ${algo} \
            --model_type dt \
            --test_trajectory_file ${test_trajectory_file} \
            --random_weights_pretrained_lm
        echo "Finished run with algorithm: $algo and seed: $seed"
        echo "------------------------"
    done
done

echo "All runs completed."
