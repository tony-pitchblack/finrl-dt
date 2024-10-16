import matplotlib.pyplot as plt

# Sample data (replace with actual data)
methods = ['a2c_cql', 'a2c_dt_lora_gpt2', 'a2c_iql', 'a2c_bc', 'a2c_dt_lora_random_weight_gpt2', 'A2C']
cumulative_returns = [46.67, 43.72, 40.26, 40.10, 38.66, 34.69]
mdds = [-9.28, -8.42, -10.12, -8.24, -9.42, -9.12]
sharpe_ratios = [2.14, 1.76, 1.84, 1.71, 1.80, 1.60]

# Bar plots for each metric
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Cumulative Return
axs[0].bar(methods, cumulative_returns, color='skyblue')
axs[0].set_title('Cumulative Return (%)')
axs[0].set_xticklabels(methods, rotation=45, ha='right')
axs[0].set_ylabel('Return (%)')

# MDD
axs[1].bar(methods, mdds, color='salmon')
axs[1].set_title('Maximum Drawdown (MDD %) [Lower is Better]')
axs[1].set_xticklabels(methods, rotation=45, ha='right')
axs[1].set_ylabel('MDD (%)')

# Sharpe Ratio
axs[2].bar(methods, sharpe_ratios, color='lightgreen')
axs[2].set_title('Sharpe Ratio [Higher is Better]')
axs[2].set_xticklabels(methods, rotation=45, ha='right')
axs[2].set_ylabel('Sharpe Ratio')

plt.tight_layout()
plt.show()
