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
from scipy import stats
from matplotlib.offsetbox import AnchoredText

from finrl.config import INDICATORS, TRAINED_MODEL_DIR

# Define calculate_metrics function here
def calculate_metrics(data):
    initial_value = data.iloc[0]
    final_value = data.iloc[-1]
    cumulative_return = (final_value - initial_value) / initial_value
    daily_returns = data.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    drawdown = (data.cummax() - data) / data.cummax()
    max_drawdown = drawdown.max()
    return cumulative_return, sharpe_ratio, max_drawdown

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
if_using_ppo = False
if_using_td3 = False
if_using_sac = False  

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
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

e_trade_gym = StockTradingEnv(df=train, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c, 
    environment=e_trade_gym) if if_using_a2c else (None, None)

df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg,
    environment=e_trade_gym) if if_using_ddpg else (None, None)

df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=trained_td3,
    environment=e_trade_gym) if if_using_td3 else (None, None)

df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_sac, environment=e_trade_gym) if if_using_sac else (None, None)

df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo, environment=e_trade_gym) if if_using_ppo else (None, None)

df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0]) if if_using_a2c else None
df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0]) if if_using_ddpg else None
df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0]) if if_using_ppo else None
df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0]) if if_using_td3 else None
df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0]) if if_using_sac else None

# Modify the merging process
result = pd.DataFrame()
if if_using_a2c: 
    df_result_a2c.columns = ['A2C']
    result = pd.merge(result, df_result_a2c, how='outer', left_index=True, right_index=True)
if if_using_ddpg: 
    df_result_ddpg.columns = ['DDPG']
    result = pd.merge(result, df_result_ddpg, how='outer', left_index=True, right_index=True)
if if_using_td3: 
    df_result_td3.columns = ['TD3']
    result = pd.merge(result, df_result_td3, how='outer', left_index=True, right_index=True)
if if_using_sac: 
    df_result_sac.columns = ['SAC']
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
    pkl_files = [f for f in os.listdir(dir_path) if f.startswith('total_asset_value_change_train') and f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        file_path = os.path.join(dir_path, pkl_file)
        data = load_pickle(file_path)
        
        # Use directory name and pkl file name (without extension) as column name
        column_name = f'{dir_name}_{os.path.splitext(pkl_file)[0]}'
        
        df = pd.DataFrame(data, columns=[column_name])
        df.index = result.index  # Assuming the dates align with the existing result dataframe
        result = pd.merge(result, df, left_index=True, right_index=True)

# Add these constants after the existing imports
TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'

# Fetch DJIA data for the entire period
df_dji = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TRAIN_END_DATE,
                         ticker_list=['^DJI']).fetch_data()  # Use '^DJI' for Dow Jones Index

df_dji = df_dji[['date', 'close']]
fst_day = df_dji['close'][0]
dji = pd.merge(df_dji['date'], df_dji['close'].div(fst_day).mul(1000000),
               how='outer', left_index=True, right_index=True).set_index('date')

# Merge DJIA into result
result = pd.merge(result, dji.rename(columns={'close': 'DJIA'}), how='outer', left_index=True, right_index=True).fillna(method='bfill')

# Function to extract experiment name from directory name
def extract_experiment_name(dir_name):
    # This regex matches everything up to the last underscore and number
    match = re.match(r'(.+)_\d+$', dir_name)
    if match:
        return match.group(1)
    return dir_name

# Group similar experiments
experiment_groups = {}
for column in result.columns:
    if column not in ['A2C', 'DDPG', 'TD3', 'SAC', 'DJIA']:
        exp_name = extract_experiment_name(column.split('_total_asset_value_change_train')[0])
        if exp_name not in experiment_groups:
            experiment_groups[exp_name] = []
        experiment_groups[exp_name].append(column)

# **Calculate metrics for all experiments and individual models**
metrics = {}
experiment_stats = {}  # Initialize experiment_stats

for exp_name, columns in experiment_groups.items():
    exp_data = result[columns]
    
    cum_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for col in columns:
        cr, sr, mdd = calculate_metrics(exp_data[col])
        cum_returns.append(cr)
        sharpe_ratios.append(sr)
        max_drawdowns.append(mdd)
    
    metrics[exp_name] = {
        'Cumulative Return': (np.mean(cum_returns), np.std(cum_returns)),
        'Sharpe Ratio': (np.mean(sharpe_ratios), np.std(sharpe_ratios)),
        'Max Drawdown': (np.mean(max_drawdowns), np.std(max_drawdowns))
    }
    
    # Store mean and std for plotting
    experiment_stats[exp_name] = {
        'mean': exp_data.mean(axis=1),
        'std': exp_data.std(axis=1)
    }

# Calculate metrics for individual models
individual_models = ['A2C', 'DDPG', 'TD3', 'SAC', 'DJIA']
for model in individual_models:
    if model in result.columns:
        cr, sr, mdd = calculate_metrics(result[model])
        metrics[model] = {
            'Cumulative Return': (cr, 0),
            'Sharpe Ratio': (sr, 0),
            'Max Drawdown': (mdd, 0)
        }

# Now start the plotting section
plt.figure(figsize=(20, 12))

# Plot DJIA
plt.plot(result.index, result['DJIA'], label="Dow Jones Index", linewidth=2, color='#1A1A1A', alpha=0.7)

# Plot experiment groups
colors = plt.cm.rainbow(np.linspace(0, 1, len(experiment_stats)))
for (exp_name, stats), color in zip(experiment_stats.items(), colors):
    mean = stats['mean']
    std = stats['std']
    
    # Plot mean
    line, = plt.plot(result.index, mean, label=exp_name, linewidth=2, color=color)
    
    # Plot error bands (mean ± 1 std)
    plt.fill_between(result.index, mean - std, mean + std, color=color, alpha=0.2)
    
    # Add metrics text box
    metrics_text = (f"{exp_name}\n"
                    f"CR: {metrics[exp_name]['Cumulative Return'][0]:.2f} ± {metrics[exp_name]['Cumulative Return'][1]:.2f}\n"
                    f"SR: {metrics[exp_name]['Sharpe Ratio'][0]:.2f} ± {metrics[exp_name]['Sharpe Ratio'][1]:.2f}\n"
                    f"MDD: {metrics[exp_name]['Max Drawdown'][0]:.2f} ± {metrics[exp_name]['Max Drawdown'][1]:.2f}")
    # Positioning the text box
    anchored_text = AnchoredText(metrics_text, loc='upper left', prop=dict(size=8), frameon=True)
    plt.gca().add_artist(anchored_text)

# Plot individual models if they exist
for model in ['A2C', 'DDPG', 'TD3', 'SAC']:
    if model in result.columns:
        plt.plot(result.index, result[model], label=model, linestyle='--', linewidth=2)
        
        # Add metrics text box for individual models
        metrics_text = (f"{model}\n"
                        f"CR: {metrics[model]['Cumulative Return'][0]:.2f}\n"
                        f"SR: {metrics[model]['Sharpe Ratio'][0]:.2f}\n"
                        f"MDD: {metrics[model]['Max Drawdown'][0]:.2f}")
        # Positioning the text box
        anchored_text = AnchoredText(metrics_text, loc='upper left', prop=dict(size=8), frameon=True)
        plt.gca().add_artist(anchored_text)

plt.title("Backtest Results from Checkpoints (with DJIA and Metrics)", fontsize=16, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Total Asset Value", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('backtest_result_from_checkpoints_with_djia_metrics_and_error_bands_train_env.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save the updated result dataframe
result.to_csv('backtest_result_from_checkpoints.csv')

print(f"Processed {len(experiment_groups)} unique experiments.")
print("Results saved as 'backtest_result_from_checkpoints_with_djia_metrics_and_error_bands_train_env.png' and 'backtest_result_from_checkpoints.csv'")