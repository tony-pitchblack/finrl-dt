import pickle
import random

# Load the A2C trajectories
with open('trajectories_a2c_100_2024-10-03_13-56-14.pkl', 'rb') as f:
    a2c_trajectories = pickle.load(f)

# Load the DDPG trajectories
with open('trajectories_ddpg_100_2024-10-04_13-56-14.pkl', 'rb') as f:
    ddpg_trajectories = pickle.load(f)

# Combine the trajectories
combined_trajectories = a2c_trajectories + ddpg_trajectories

# Update the 'name' field for all trajectories
for trajectory in combined_trajectories:
    trajectory['name'] = 'a2c+ddpg'

# Shuffle the combined trajectories
random.shuffle(combined_trajectories)

# Save the mixed and shuffled trajectories
output_filename = 'trajectories_a2c+ddpg_mixed_shuffled.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump(combined_trajectories, f)

print(f"Mixed and shuffled trajectories saved to {output_filename}")