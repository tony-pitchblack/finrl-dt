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

algorithms = ['a2c', 'ddpg', 'ppo', 'td3', 'sac']

for current_algo in algorithms:
    # Reset all algorithms to False
    if_using_a2c = False
    if_using_ddpg = False
    if_using_ppo = False
    if_using_td3 = False
    if_using_sac = False

    # Set the current algorithm to True
    if current_algo == 'a2c':
        if_using_a2c = True
    elif current_algo == 'ddpg':
        if_using_ddpg = True
    elif current_algo == 'ppo':
        if_using_ppo = True
    elif current_algo == 'td3':
        if_using_td3 = True
    elif current_algo == 'sac':
        if_using_sac = True

    # Reset algos_included for each iteration
    algos_included = ''

    print(f"Running with {current_algo.upper()} set to True")

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

    e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_a2c, 
        environment = e_trade_gym) if if_using_a2c else (None, None)

    df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
        model=trained_ddpg,
        environment = e_trade_gym) if if_using_ddpg else (None, None)

    df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
        model=trained_td3,
        environment = e_trade_gym) if if_using_td3 else (None, None)

    df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
        model=trained_sac, environment = e_trade_gym) if if_using_sac else (None, None)

    df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
        model=trained_ppo, environment = e_trade_gym) if if_using_ppo else (None, None)

    df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0]) if if_using_a2c else None
    df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0]) if if_using_ddpg else None
    df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0]) if if_using_ppo else None
    df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0]) if if_using_td3 else None
    df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0]) if if_using_sac else None

    # Modify the merging process
    result = pd.DataFrame()
    if if_using_a2c: 
        algos_included += '_a2c'
        df_result_a2c.columns = ['A2C_' + col for col in df_result_a2c.columns]
        result = pd.merge(result, df_result_a2c, how='outer', left_index=True, right_index=True)
    if if_using_ddpg:
        algos_included += '_ddpg'
        df_result_ddpg.columns = ['DDPG_' + col for col in df_result_ddpg.columns]
        result = pd.merge(result, df_result_ddpg, how='outer', left_index=True, right_index=True)
    if if_using_td3: 
        algos_included += '_td3'
        df_result_td3.columns = ['TD3_' + col for col in df_result_td3.columns]
        result = pd.merge(result, df_result_td3, how='outer', left_index=True, right_index=True)
    if if_using_sac: 
        algos_included += '_sac'
        df_result_sac.columns = ['SAC_' + col for col in df_result_sac.columns]
        result = pd.merge(result, df_result_sac, how='outer', left_index=True, right_index=True)
    if if_using_ppo:
        algos_included += '_ppo'
        df_result_ppo.columns = ['PPO_' + col for col in df_result_ppo.columns]
        result = pd.merge(result, df_result_ppo, how='outer', left_index=True, right_index=True)

    # Function to load pickle files
    def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # Get all directories under ./checkpoints/
    checkpoint_dirs = [d for d in os.listdir('./checkpoints') if os.path.isdir(os.path.join('./checkpoints', d))]

    # Process each checkpoint directory
    for dir_name in checkpoint_dirs:
        dir_path = os.path.join('./checkpoints', dir_name)
        pkl_files = [f for f in os.listdir(dir_path) if f.startswith('total_asset_value_change_test') and f.endswith('.pkl')]
        
        for pkl_file in pkl_files:
            file_path = os.path.join(dir_path, pkl_file)
            data = load_pickle(file_path)
            data = data[:335]
            
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
    if if_using_ppo: col_name.append('PPO')
    col_name.extend(result.columns[len(col_name):])  # Add all other column names
    result.columns = col_name

    # Add these constants after the existing imports
    TEST_START_DATE = '2020-07-01'
    TEST_END_DATE = '2021-10-29'

    # After processing the checkpoint directories and before plotting

    # Fetch DJIA data for the test period
    df_dji = YahooDownloader(start_date=TEST_START_DATE,
                             end_date=TEST_END_DATE,
                             ticker_list=['dji']).fetch_data()

    df_dji = df_dji[['date','close']]
    fst_day = df_dji['close'][0]
    dji = pd.merge(df_dji['date'], df_dji['close'].div(fst_day).mul(1000000), 
                   how='outer', left_index=True, right_index=True).set_index('date')

    result = pd.merge(result, dji, how='outer', left_index=True, right_index=True).fillna(method='bfill')
    result = result.rename(columns={'close': 'DJIA'})

    # Add these new control variables
    include_ensemble = False  # Set to False to exclude ensemble experiments
    exclude_algo_experiments = True  # Set to True to exclude individual algo experiments based on if_using_... flags

    if include_ensemble:
        algos_included += '_ensemble'

    # Function to extract experiment name from directory name
    def extract_experiment_name(dir_name):
        # This regex matches everything up to the last underscore and number
        match = re.match(r'(.+)_\d+$', dir_name)
        if match:
            return match.group(1)
        return dir_name

    # Function to check if an experiment should be included
    def should_include_experiment(exp_name):
        if not include_ensemble and 'ensemble' in exp_name.lower():
            return False
        if exclude_algo_experiments:
            algo_flags = {
                'a2c': if_using_a2c,
                'ddpg': if_using_ddpg,
                'ppo': if_using_ppo,
                'td3': if_using_td3,
                'sac': if_using_sac
            }
            for algo, flag in algo_flags.items():
                if algo in exp_name.lower() and not flag:
                    return False
        return True

    # Group similar experiments
    experiment_groups = {}
    for column in result.columns:
        if column not in ['A2C', 'DDPG', 'TD3', 'SAC', 'DJIA']:
            exp_name = extract_experiment_name(column.split('_total_asset_value_change_test')[0])
            if should_include_experiment(exp_name):
                if exp_name not in experiment_groups:
                    experiment_groups[exp_name] = []
                experiment_groups[exp_name].append(column)

    # Calculate mean and std for each experiment group
    experiment_stats = {}
    for exp_name, columns in experiment_groups.items():
        exp_data = result[columns]
        exp_mean = exp_data.mean(axis=1)
        exp_std = exp_data.std(axis=1)
        experiment_stats[exp_name] = {'mean': exp_mean, 'std': exp_std}

    # Modify the plotting section
    plt.figure(figsize=(15, 8))

    # Plot DJIA
    plt.plot(result.index, result['DJIA'], label="Dow Jones Index", linewidth=2, color='#1A1A1A', alpha=0.7)

    # Plot experiment groups
    for exp_name, stats in experiment_stats.items():
        mean = stats['mean']
        std = stats['std']
        
        # Plot mean
        line, = plt.plot(result.index, mean, label=exp_name, linewidth=2)
        
        # Plot error bands (mean ± 1 std)
        plt.fill_between(result.index, mean - std, mean + std, color=line.get_color(), alpha=0.2)

    # Plot individual models if they exist
    for model in ['A2C', 'DDPG', 'TD3', 'SAC']:
        if model in result.columns:
            plt.plot(result.index, result[model], label=model, linestyle='--', linewidth=2)

    plt.title("Backtest Results from Checkpoints (with DJIA)", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Asset Value", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(f'backtest_result_from_checkpoints_with_djia_and_error_bands_test_env_{current_algo}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Results saved as 'backtest_result_from_checkpoints_with_djia_and_error_bands_test_env_{current_algo}.png'")
    print("----------------------------------------")
