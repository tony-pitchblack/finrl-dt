#!/bin/bash
# Auto-generated run script for SAC

conda activate finrl-dt

export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

seeds=(20742 55230 85125 96921 67851)
gpu=0 # GPU identification number

drl_alg=sac # The deep reinforcement learning algorithm we are targeting to clone the behavior
model_type=dt

# Training parameters
lr=1e-3
weight_decay=1e-5 # For AdamW optimizer
dropout=0.1
warmup_steps=2500
num_steps=1000 # Total number of training steps; i.e., how many times we call env.step() for training

# Environment parameters
env=stock_trading
sample_ratio=1
K=20  # Context length for the decision transformer model
dataset_path=data/train_sac_trajectory_2024-10-12_14-37-41.pkl # Path to the trajectory data
test_trajectory_file=data/test_sac_trajectory_2024-10-12_14-37-44.pkl

# Device
device='cuda' # or 'cpu'

# Pretrained language model
pretrained_lm="/home/gridsan/syun/gpt2_model" # Path to the downloaded GPT-2 model

use_pretrained_lm=true
lora=true

for seed in "${seeds[@]}"; do
    exp_name="${drl_alg}_${model_type}_lora_gpt2_${seed}"
    outdir="/home/gridsan/syun/finrl-dt/checkpoints/${exp_name}"

    # Run the experiment
    CUDA_VISIBLE_DEVICES=${gpu} python ~/finrl-dt/experiment.py \
        --device ${device} \
        --env ${env} \
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
        --drl_alg ${drl_alg} \
        --model_type ${model_type} \
        --test_trajectory_file ${test_trajectory_file}
done
