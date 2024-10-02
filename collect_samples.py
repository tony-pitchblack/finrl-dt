import numpy as np
import pandas as pd
import pickle
import gymnasium as gym
from tqdm import tqdm
from stable_baselines3 import A2C
from finrl.config import INDICATORS, TRAINED_MODEL_DIR

train = pd.read_csv('train_data.csv')
train = train.set_index(train.columns[0])
train.index.names = ['']

model_a2c = A2C.load("agent_a2c.zip")

print(train.head())
print(train.close.values)

def collect_trajectories_a2c(env, model, num_episodes=100):
    trajectories = []

    for episode in tqdm(range(num_episodes)):
        observations = []
        actions = []
        rewards = []
        dones = []

        state, _ = env.reset()
        done = False

        while not done:
            # Use the trained A2C model to select actions
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

        trajectories.append(trajectory)

        print(f"Collected trajectory {episode + 1}/{num_episodes}")

    return trajectories

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

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


env = StockTradingEnv(df = train, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)


# Collect the trajectories using the trained A2C agent
num_episodes = 1  # You can adjust this number
trajectories_a2c = collect_trajectories_a2c(env, model_a2c, num_episodes=num_episodes)

# Save the trajectories to a pickle file
# current time 
from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

with open(f'trajectories_a2c_{num_episodes}_{current_time}.pkl', 'wb') as f:
    pickle.dump(trajectories_a2c, f)