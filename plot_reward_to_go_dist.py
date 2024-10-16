import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

model = 'ddpg'

# Load the trajectories from the pickle file
with open('trajectories_a2c_1_2024-10-06_14-19-28.pkl', 'rb') as f:
    trajectories = pickle.load(f)

# Calculate total reward for each trajectory
total_rewards = []
for trajectory in tqdm(trajectories, desc="Calculating total rewards"):
    # print(trajectory['observations'][100][0])
    print(trajectory['rewards'])
    total_reward = np.sum(trajectory['rewards'])
    # print(total_reward)
    if total_reward < 370:
        print("lower than 370: ", total_reward)
    else:
        print("higher than 370: ", total_reward)
    total_rewards.append(total_reward)

# Create a line plot
plt.figure(figsize=(10, 6))
sns.kdeplot(total_rewards, shade=True)
plt.title('Distribution of Total Rewards per Episode')
plt.xlabel('Total Reward')
plt.ylabel('Density')

# Add summary statistics
mean_reward = np.mean(total_rewards)
median_reward = np.median(total_rewards)
plt.axvline(mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
plt.axvline(median_reward, color='g', linestyle='--', label=f'Median: {median_reward:.2f}')
plt.legend()

# Save the plot
plt.savefig(f'reward_distribution_{model}.png')
plt.close()

# Print summary statistics
print(f"Number of trajectories: {len(total_rewards)}")
print(f"Mean total reward: {mean_reward:.2f}")
print(f"Median total reward: {median_reward:.2f}")
print(f"Min total reward: {np.min(total_rewards):.2f}")
print(f"Max total reward: {np.max(total_rewards):.2f}")

# Calculate and print the percentage of positive rewards
positive_rewards = sum(reward > 0 for reward in total_rewards)
percentage_positive = (positive_rewards / len(total_rewards)) * 100
print(f"Percentage of positive rewards: {percentage_positive:.2f}%")