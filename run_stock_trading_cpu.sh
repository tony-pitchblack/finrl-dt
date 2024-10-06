#!/bin/bash
# run with: source run_stock_trading.sh "test_run" 123 0   
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

# Model parameters
model_type=dt
lr=1e-4
lmlr=1e-5
weight_decay=1e-5
dropout=0.1
warmup_steps=2500
num_steps_per_iter=1
max_iters=40
num_eval_episodes=20


# Environment parameters
env=stock_trading
dataset=your_dataset_name
sample_ratio=1
K=20  # Context length
state_dim=291
act_dim=29
dataset_path="trajectories_a2c_100_2024-10-03_13-56-14.pkl"

# Device
device="cpu"

# Pretrained language model
pretrained_lm="gpt2"
# pretrained_lm="/state/partition1/user/syun/gpt2_model"
# pretrained_lm="/home/gridsan/syun/gpt2_model"

# Positional arguments from command line
description=${1}
seed=${2}
gpu=${3}

exp_name="test run"
drl_algo="a2c"

# Construct description and output directory
description="${pretrained_lm}_pretrained-ratio=${sample_ratio}_${description}"
outdir="checkpoints/${env}_${dataset}_${description}_${seed}"

# Run the experiment
CUDA_VISIBLE_DEVICES=${gpu} ~/gits/FinRL-Tutorials/.conda/bin/python experiment.py \
    --device ${device} \
    --env ${env} \
    --dataset ${dataset} \
    --dataset_path ${dataset_path} \
    --seed ${seed} \
    --K ${K} \
    --learning_rate ${lr} \
    --lm_learning_rate ${lmlr} \
    --num_steps_per_iter ${num_steps_per_iter} \
    --weight_decay ${weight_decay} \
    --max_iters ${max_iters} \
    --num_eval_episodes ${num_eval_episodes} \
    --sample_ratio ${sample_ratio} \
    --warmup_steps ${warmup_steps} \
    --pretrained_lm ${pretrained_lm} \
    --outdir ${outdir} \
    --dropout ${dropout} \
    --description ${description} \
    --mlp_embedding \
    --adapt_mode \
    --adapt_embed \
    --lora \
    --exp_name ${exp_name} \
    --drl_algo ${drl_algo}
