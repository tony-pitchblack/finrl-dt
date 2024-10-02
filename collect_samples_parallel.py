import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import multiprocessing
from datetime import datetime
from finrl.config import INDICATORS, TRAINED_MODEL_DIR

def collect_single_trajectory(args):
    # Unpack the arguments
    model_path, env_kwargs, train_data_file = args

    # Load the data within the process
    import pandas as pd
    train_data = pd.read_csv(train_data_file)
    train_data = train_data.set_index(train_data.columns[0])
    train_data.index.names = ['']

    # Initialize the environment within the process
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    env = StockTradingEnv(
        df=train_data,
        turbulence_threshold=70,
        risk_indicator_col='vix',
        **env_kwargs
    )

    # Load the model within the process
    from stable_baselines3 import A2C
    model_a2c = A2C.load(model_path)

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
    # Load the training data
    train_data_file = 'train_data.csv'
    train = pd.read_csv(train_data_file)
    train = train.set_index(train.columns[0])
    train.index.names = ['']

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

    model_path = TRAINED_MODEL_DIR + "/agent_a2c"

    # Number of trajectories to collect
    num_episodes = 100  # Adjust this number as needed

    # Prepare arguments for each process
    args_list = [
        (model_path, env_kwargs, train_data_file)
        for _ in range(num_episodes)
    ]

    # Detect the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Detected {num_cores} CPU cores.")

    # Use multiprocessing to collect trajectories in parallel
    # Number of worker processes is set to the number of CPU cores
    with multiprocessing.Pool(processes=num_cores) as pool:
        trajectories = list(tqdm(
            pool.imap(collect_single_trajectory, args_list),
            total=num_episodes
        ))

    # Save the trajectories to a pickle file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'trajectories_a2c_{num_episodes}_{current_time}.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"Collected {num_episodes} trajectories and saved to '{output_filename}'")
