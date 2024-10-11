#!/bin/bash
# run with: source run_stock_trading.sh "test_run" 123 0   
conda activate finrl-dt

export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

seeds=(20742 55230 85125 96921 67851)
gpu=0 # gpu identification number

drl_alg=a2c # the deep reinforcement learning algorithm of which we are targeting to clone the behavior
model_type=dt

# Training parameters
lr=1e-3
weight_decay=1e-5 # for AdamW optimizer
dropout=0.1
warmup_steps=2500
num_steps=1000 # total number of training steps; i.e., how many times we call env.step() for training

# Environment parameters
env=stock_trading
dataset=your_dataset_name
sample_ratio=1
K=20  # Context length - this is for the decision transformer model; Seeing the K number of states in the past to make a prediction on next action
dataset_path=train_trajectories_a2c_1_2024-10-11_16-03-20.pkl # path to the trajectory data; It's a list of dict, where each dict contains keys like "observations", "actions", "rewards", and "terminals"
test_trajectory_file=test_trajectories_a2c_1_2024-10-11_16-04-31.pkl

# Device
device='cuda' # or 'cpu'

# Pretrained language model
# pretrained_lm="gpt2" # this will trigger auto-downloading the gpt2 model from the Hugging Face model hub
pretrained_lm="/home/gridsan/syun/gpt2_model" # or, we can simply use the path to the downloaded gpt2 model

use_pretrained_lm=true
lora=true

for seed in "${seeds[@]}"; do
    exp_name="${drl_alg}_${model_type}_lora_gpt2_${seed}"
    outdir="/home/gridsan/syun/finrl-dt/checkpoints/${exp_name}"

    # Run the experiment
    CUDA_VISIBLE_DEVICES=${gpu} python experiment.py \
        --device ${device} \
        --env ${env} \
        --dataset ${dataset} \
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