import sys
import os
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
if_using_ddpg = False
# if_using_ppo = True
# if_using_td3 = True
# if_using_sac = True

if if_using_a2c:
    trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")
if if_using_ddpg:
    trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg")
# trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
# trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
# trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

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
    # model=trained_a2c, 
    model=trained_a2c,
    environment = e_trade_gym) if if_using_a2c else (None, None)

df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0]) if if_using_a2c else None

# Function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Get all directories in the current path
directories = [d for d in os.listdir('checkpoints') if os.path.isdir(os.path.join('checkpoints', d))]

result = pd.DataFrame()
if if_using_a2c: result = pd.merge(result, df_result_a2c, how='outer', left_index=True, right_index=True)

# Load and merge data from all directories
for directory in directories:
    file_path = os.path.join('./', directory, 'total_asset_value_change_eval_1500000.pkl')
    if os.path.exists(file_path):
        data = load_pickle(file_path)
        df = pd.DataFrame(data, columns=[directory])
        df.index = result.index  # Assuming the dates align with the existing result dataframe
        result = pd.merge(result, df, left_index=True, right_index=True)

# Create column names list
col_name = []
if if_using_a2c and 'A2C' in result.columns:
    col_name.append('A2C')
col_name.extend([d for d in directories if d in result.columns])

# Ensure col_name matches the number of columns in result
if len(col_name) != len(result.columns):
    print(f"Warning: Mismatch between column names ({len(col_name)}) and actual columns ({len(result.columns)})")
    print("Columns in result:", result.columns)
    print("Generated column names:", col_name)
    
    # Use the actual column names if there's a mismatch
    col_name = list(result.columns)

result.columns = col_name

# Plotting
plt.figure(figsize=(15, 8))
for column in result.columns:
    if column == 'A2C':
        plt.plot(result.index, result[column], label=column, linestyle='--', linewidth=2)
    else:
        plt.plot(result.index, result[column], label=column)

plt.title("Backtest Results for Different Experiments")
plt.xlabel("Date")
plt.ylabel("Total Asset Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('backtest_result_multiple_experiments.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the updated result dataframe
result.to_csv('updated_backtest_result_multiple_experiments.csv')

print(f"Processed {len(directories)} directories.")
print("Results saved as 'backtest_result_multiple_experiments.png' and 'updated_backtest_result_multiple_experiments.csv'")