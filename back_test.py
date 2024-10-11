import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.config import INDICATORS, TRAINED_MODEL_DIR


train = pd.read_csv('train_data.csv')
trade = pd.read_csv('test_data.csv')

# If you are not using the data generated from part 1 of this tutorial, make sure 
# it has the columns and index in the form that could be make into the environment. 
# Then you can comment and skip the following lines.
train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

if_using_a2c = True
if_using_ddpg = True
# if_using_ppo = True
if_using_td3 = True
if_using_sac = True

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
# trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


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

e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c, 
    # model=trained_ddpg,
    environment = e_trade_gym) if if_using_a2c else (None, None)

df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg,
    environment = e_trade_gym) if if_using_ddpg else (None, None)

df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=trained_td3,
    environment = e_trade_gym) if if_using_td3 else (None, None)

df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_sac, environment = e_trade_gym) if if_using_sac else (None, None)

df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0]) if if_using_a2c else None
df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0]) if if_using_ddpg else None
# df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0]) if if_using_ppo else None
df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0]) if if_using_td3 else None
df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0]) if if_using_sac else None

# Modify the merging process
result = pd.DataFrame()
if if_using_a2c: 
    df_result_a2c.columns = ['A2C_' + col for col in df_result_a2c.columns]
    result = pd.merge(result, df_result_a2c, how='outer', left_index=True, right_index=True)
if if_using_ddpg: 
    df_result_ddpg.columns = ['DDPG_' + col for col in df_result_ddpg.columns]
    result = pd.merge(result, df_result_ddpg, how='outer', left_index=True, right_index=True)
if if_using_td3: 
    df_result_td3.columns = ['TD3_' + col for col in df_result_td3.columns]
    result = pd.merge(result, df_result_td3, how='outer', left_index=True, right_index=True)
if if_using_sac: 
    df_result_sac.columns = ['SAC_' + col for col in df_result_sac.columns]
    result = pd.merge(result, df_result_sac, how='outer', left_index=True, right_index=True)

# Function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Get all directories under ./checkpoints/
checkpoint_dirs = [d for d in os.listdir('./checkpoints') if os.path.isdir(os.path.join('./checkpoints', d))]

# Process each checkpoint directory
for dir_name in checkpoint_dirs:
    dir_path = os.path.join('./checkpoints', dir_name)
    pkl_files = [f for f in os.listdir(dir_path) if f.startswith('total_asset_value_change') and f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        file_path = os.path.join(dir_path, pkl_file)
        data = load_pickle(file_path)
        
        # Use directory name and pkl file name (without extension) as column name
        column_name = f'{dir_name}_{os.path.splitext(pkl_file)[0]}'
        
        df = pd.DataFrame(data, columns=[column_name])
        df.index = result.index  # Assuming the dates align with the existing result dataframe
        result = pd.merge(result, df, left_index=True, right_index=True)

# Create column names list
col_name = []
if if_using_a2c: col_name.append('A2C')
if if_using_ddpg: col_name.append('DDPG')
if if_using_td3: col_name.append('TD3')
if if_using_sac: col_name.append('SAC')
col_name.extend(result.columns[len(col_name):])  # Add all other column names
result.columns = col_name

# Plotting
plt.figure(figsize=(15, 8))
for column in result.columns:
    if column.startswith(('A2C_', 'DDPG_', 'TD3_', 'SAC_')):
        plt.plot(result.index, result[column], label=column.split('_')[0], linestyle='--', linewidth=2)
    else:
        plt.plot(result.index, result[column], label=column)

plt.title("Backtest Results from Checkpoints")
plt.xlabel("Date")
plt.ylabel("Total Asset Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('backtest_result_from_checkpoints.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the updated result dataframe
result.to_csv('backtest_result_from_checkpoints.csv')

print(f"Processed {len(checkpoint_dirs)} checkpoint directories.")
print("Results saved as 'backtest_result_from_checkpoints.png' and 'backtest_result_from_checkpoints.csv'")
