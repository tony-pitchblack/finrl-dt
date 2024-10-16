import os
# Set environment variables to limit threads for reproducibility and performance
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
np.random.seed(random_seed)

def collect_single_trajectory(env_kwargs, model, train_data):
    """
    Collects a single trajectory using the provided model and environment parameters.

    Args:
        env_kwargs (dict): Configuration parameters for the environment.
        model (StableBaselines3 RL model): Trained RL model to generate actions.
        train_data (pd.DataFrame): Historical training data for the environment.

    Returns:
        dict: A dictionary containing observations, actions, rewards, and terminal flags.
    """
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
        # Use the trained RL model to select actions
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

    # Number of trajectories to collect per model
    episodes_per_model = 1  # Keep this as 1

    # Define downsampling ratios (10%, 20%, ..., 100%)
    downsampling_ratios = [0.1 * i for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]

    # Ensure the 'data' directory exists
    data_dir = 'data_downsamples'
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

        ensemble_trajectories = []

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

            # Collect one trajectory for the current model
            trajectory = collect_single_trajectory(env_kwargs, model, data)
            
            # Initialize current time for unique filenames
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            if train_or_test == 'train':
                # Apply downsampling to the trajectory at specified ratios
                for ratio in downsampling_ratios:
                    # Calculate the number of steps to retain
                    n_steps = int(ratio * len(trajectory['observations']))
                    if n_steps == 0:
                        print(f"Ratio {int(ratio*100)}% is too small for the trajectory length. Skipping.")
                        continue

                    # Create the downsampled trajectory
                    downsampled_trajectory = {
                        'observations': trajectory['observations'][:n_steps],
                        'actions': trajectory['actions'][:n_steps],
                        'rewards': trajectory['rewards'][:n_steps],
                        'terminals': trajectory['terminals'][:n_steps]
                    }

                    # Define the output filename with downsampling ratio
                    ratio_percentage = int(ratio * 100)
                    output_filename = f'{train_or_test}_{model_choice}_trajectory_{ratio_percentage}percent_{current_time}.pkl'
                    output_path = os.path.join(data_dir, output_filename)

                    # Save the downsampled trajectory
                    with open(output_path, 'wb') as f:
                        pickle.dump([downsampled_trajectory], f)
                    
                    print(f"Saved downsampled trajectory ({ratio_percentage}%) for {model_choice.upper()} on {train_or_test} data to '{output_path}'")
                    
                    # Add to ensemble trajectories
                    ensemble_trajectories.append(downsampled_trajectory)
            else:
                # For 'test' dataset, save the full trajectory without downsampling
                output_filename = f'{train_or_test}_{model_choice}_trajectory_full_{current_time}.pkl'
                output_path = os.path.join(data_dir, output_filename)
                with open(output_path, 'wb') as f:
                    pickle.dump([trajectory], f)
                
                print(f"Saved full trajectory for {model_choice.upper()} on {train_or_test} data to '{output_path}'")
                
                # Add to ensemble trajectories
                ensemble_trajectories.append(trajectory)

        if train_or_test == 'train':
            # Save the ensemble trajectories (all downsampled trajectories)
            ensemble_filename = f'{train_or_test}_ensemble_trajectories_{len(model_choices)}_{current_time}.pkl'
            ensemble_path = os.path.join(data_dir, ensemble_filename)
            with open(ensemble_path, 'wb') as f:
                pickle.dump(ensemble_trajectories, f)

            print(f"Saved ensemble of {len(ensemble_trajectories)} downsampled trajectories (one from each model at each ratio) on {train_or_test} data to '{ensemble_path}'")
        else:
            # Save the ensemble trajectories (all full trajectories)
            ensemble_filename = f'{train_or_test}_ensemble_trajectories_{len(model_choices)}_{current_time}.pkl'
            ensemble_path = os.path.join(data_dir, ensemble_filename)
            with open(ensemble_path, 'wb') as f:
                pickle.dump(ensemble_trajectories, f)

            print(f"Saved ensemble of {len(model_choices)} trajectories (one from each model) on {train_or_test} data to '{ensemble_path}'")
