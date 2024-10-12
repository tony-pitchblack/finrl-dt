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
from stable_baselines3 import A2C, DDPG, TD3, PPO, SAC

random_seed = 21102

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
    # Define the list of models and datasets
    model_choices = ['a2c', 'ddpg', 'td3', 'ppo', 'sac']
    dataset_choices = ['train', 'test']

    # Number of trajectories to collect
    total_episodes = 1  # Adjust this number as needed

    # Ensure the 'data' directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for train_or_test in dataset_choices:
        # Load the dataset
        data_file = f'{train_or_test}_data.csv'
        data = pd.read_csv(data_file)
        data = data.set_index(data.columns[0])
        data.index.names = ['']

        # Define environment parameters
        stock_dimension = len(data.tic.unique())
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
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }

        for model_choice in model_choices:
            # Load the model
            model_path = os.path.join(TRAINED_MODEL_DIR, f"agent_{model_choice}")
            if model_choice == 'a2c':
                model = A2C.load(model_path)
            elif model_choice == 'ddpg':
                model = DDPG.load(model_path)
            elif model_choice == 'td3':
                model = TD3.load(model_path)
            elif model_choice == 'ppo':
                model = PPO.load(model_path)
            elif model_choice == 'sac':
                model = SAC.load(model_path)
            else:
                raise ValueError(f"Unknown model choice: {model_choice}")

            # Collect trajectories
            trajectories = []
            for _ in tqdm(range(total_episodes), desc=f"Collecting Trajectories for {model_choice.upper()} on {train_or_test} data"):
                trajectory = collect_single_trajectory(env_kwargs, model, data)
                trajectories.append(trajectory)

            # Save the trajectories to a pickle file
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f'{train_or_test}_trajectories_{model_choice}_{total_episodes}_{current_time}.pkl'
            output_path = os.path.join(data_dir, output_filename)
            with open(output_path, 'wb') as f:
                pickle.dump(trajectories, f)

            print(f"Collected {total_episodes} trajectories using {model_choice.upper()} model on {train_or_test} data and saved to '{output_path}'")
