#!/bin/bash
# run with: source run_stock_trading.sh "test_run" 123 0   
conda activate finrl-dt

export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

seed=11102

# Device
device='cuda' #'cpu' or 'cuda'
gpu=0 # gpu identification number

drl_alg=a2c # the deep reinforcement learning algorithm of which we are targeting to clone the behavior
model_type=dt

# Training parameters
lr=1e-4
weight_decay=1e-5 # for AdamW optimizer
dropout=0.1
warmup_steps=2500
num_steps=75500 # total number of training steps; i.e., how many times we call env.step() for training
batch_size=64

# Environment parameters
env=stock_trading
dataset=your_dataset_name
sample_ratio=1
K=20  # Context length - this is for the decision transformer model; Seeing the K number of states in the past to make a prediction on next action
dataset_path="./trajectories_a2c_1_2024-10-06_14-19-28.pkl" # path to the trajectory data; It's a list of dict, where each dict contains keys like "observations", "actions", "rewards", and "terminals"


# Pretrained language model
pretrained_lm="gpt2" # this will trigger auto-downloading the gpt2 model from the Hugging Face model hub
# pretrained_lm="/home/gridsan/syun/gpt2_model" # or, we can simply use the path to the downloaded gpt2 model

lora=true
exp_name="${drl_alg}_${model_type}_lora_${lora}_${pretrained_lm}_${seed}"

outdir="checkpoints/${exp_name}"

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
    --batch_size ${batch_size}
