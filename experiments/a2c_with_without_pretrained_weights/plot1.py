import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load the A2C results from CSV
result = pd.read_csv('updated_backtest_result_multiple_targets_a2c_random_weights.csv', index_col=0)
result.index = pd.to_datetime(result.index)

# Load the specific pickle files
path_prefix = './experiments/a2c_with_without_pretrained_weights'  # Fixed typo in 'pretrained'
with_pretrained_path = f'{path_prefix}/with/total_asset_value_list_1500000_2024-10-03_15-50-02_a2c_with_pretrained.pkl'
without_pretrained_path = f'{path_prefix}/without/total_asset_value_list_1500000_2024-10-03_16-17-35_a2c_random_weights.pkl'

try:
    with_pretrained = load_pickle(with_pretrained_path)
    without_pretrained = load_pickle(without_pretrained_path)
except FileNotFoundError as e:
    print(f"Error: File not found. {e}")
    print("Please check if the file paths are correct and the files exist.")
    exit(1)
# Create DataFrames for the loaded data
df_with_pretrained = pd.DataFrame(with_pretrained, columns=['A2C_with_pretrained'])
df_without_pretrained = pd.DataFrame(without_pretrained, columns=['A2C_without_pretrained'])

# Set the index for the new DataFrames to match the result DataFrame
df_with_pretrained.index = result.index
df_without_pretrained.index = result.index

# Merge the new data with the existing result DataFrame
result['A2C_with_pretrained'] = df_with_pretrained['A2C_with_pretrained']
result['A2C_without_pretrained'] = df_without_pretrained['A2C_without_pretrained']

# Plotting
plt.figure(figsize=(15, 8))
plt.plot(result.index, result['A2C'], label='A2C (Original)', linestyle='--', linewidth=2)
plt.plot(result.index, result['A2C_with_pretrained'], label='Decision Transformer + LoRA trained with A2C trajectories with GPT-2 Pretrained Weights', linewidth=2)
plt.plot(result.index, result['A2C_without_pretrained'], label='Decision Transformer + LoRA trained with A2C trajectories with GPT-2 Weights Randomly Initialized', linewidth=2)

plt.title("Backtest Results: A2C vs. With/Without Pretrained Weights")
plt.xlabel("Date")
plt.ylabel("Total Asset Value")
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('backtest_result_a2c_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the updated result dataframe
result.to_csv('updated_backtest_result_a2c_comparison.csv')

print("Results saved as 'backtest_result_a2c_comparison.png' and 'updated_backtest_result_a2c_comparison.csv'")