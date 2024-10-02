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
import multiprocessing
from datetime import datetime
from finrl.config import INDICATORS, TRAINED_MODEL_DIR

def init_worker(model, train_data):
    # Set global variables for the worker processes
    global model_a2c_global
    global train_data_global
    model_a2c_global = model
    train_data_global = train_data

def collect_single_trajectory(env_kwargs):
    # Use the global model and data
    model_a2c = model_a2c_global
    train_data = train_data_global

    # Initialize the environment within the process
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
        action, _ = model_a2c.predict(state, deterministic=True)

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
    train_data_file = 'train_data.csv'
    train = pd.read_csv(train_data_file)
    train = train.set_index(train.columns[0])
    train.index.names = ['']

    # Load the model once
    from stable_baselines3 import A2C
    model_path = TRAINED_MODEL_DIR + "/agent_a2c"
    model_a2c = A2C.load(model_path)

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
    total_episodes = 100  # Adjust this number as needed

    # Detect the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Detected {num_cores} CPU cores.")

    # Limit the number of worker processes to avoid resource exhaustion
    max_workers = num_cores  # Adjust 16 to a suitable number if needed
    print(f"Using {max_workers} worker processes.")

    # Prepare the environment arguments list
    env_args_list = [env_kwargs for _ in range(total_episodes)]

    # Use a persistent Pool and initialize with shared resources
    with multiprocessing.Pool(processes=max_workers, initializer=init_worker, initargs=(model_a2c, train)) as pool:
        trajectories = list(tqdm(
            pool.imap_unordered(collect_single_trajectory, env_args_list),
            total=total_episodes,
            desc="Collecting Trajectories"
        ))

    # Save the trajectories to a pickle file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'trajectories_a2c_{total_episodes}_{current_time}.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"Collected {total_episodes} trajectories and saved to '{output_filename}'")
