import os
# Set environment variables to limit threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from datetime import datetime
from finrl.config import INDICATORS, TRAINED_MODEL_DIR

random_seed = 20742

def collect_single_trajectory(env_kwargs, model, train_data):
    # Initialize the environment
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    env = StockTradingEnv(
        df=train_data,
        turbulence_threshold=70,
        risk_indicator_col='vix',
        **env_kwargs
    )

    observations = []
    actions = []
    rewards = []
    dones = []

    state, _ = env.reset()
    done = False

    while not done:
        # Use the trained A2C model to select actions
        model.set_random_seed(random_seed)
        action, _ = model.predict(state, deterministic=True)
        
        next_state, reward, done, truncated, info = env.step(action)

        observations.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        state = next_state

    trajectory = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminals': np.array(dones)
    }

    return trajectory

if __name__ == "__main__":
    # Load the training data once
    train_or_test = 'test' # or 'train'
    train_data_file = f'{train_or_test}_data.csv'
    train = pd.read_csv(train_data_file)
    train = train.set_index(train.columns[0])
    train.index.names = ['']

    # Load the model once
    from stable_baselines3 import A2C, DDPG
    model_choice = 'a2c'    
    
    model_path = TRAINED_MODEL_DIR + f"/agent_{model_choice}"
    if model_choice == 'a2c':
        model = A2C.load(model_path)
    elif model_choice == 'ddpg':
        model = DDPG.load(model_path)

    # Define environment parameters
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,  # Define your technical indicators
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    # Number of trajectories to collect
    total_episodes = 1  # Adjust this number as needed

    # Collect trajectories sequentially
    trajectories = []
    for _ in tqdm(range(total_episodes), desc="Collecting Trajectories"):
        trajectory = collect_single_trajectory(env_kwargs, model, train)
        trajectories.append(trajectory)

    # Save the trajectories to a pickle file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'{train_or_test}_trajectories_{model_choice}_{total_episodes}_{current_time}.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"Collected {total_episodes} trajectories using {model_choice.upper()} model and saved to '{output_filename}'")