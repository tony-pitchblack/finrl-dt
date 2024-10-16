import pickle
import hashlib
import numpy as np

# output_filename = 'trajectories_ddpg_100_2024-10-04_13-56-14.pkl'
output_filename = 'trajectories_a2c_100_2024-10-03_13-56-14.pkl'

with open(output_filename, 'rb') as f:
    trajectories = pickle.load(f)

total_trajectories = len(trajectories)
print(f"Total number of trajectories: {total_trajectories}")

def hash_trajectory(traj):
    traj_bytes = b''
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        array = traj[key]
        if array.dtype == np.bool_:
            array_bytes = array.astype(np.uint8).tobytes()
        else:
            array = np.round(array, decimals=5)
            array_bytes = array.astype(np.float64).tobytes()
        traj_bytes += array_bytes
    return hashlib.sha256(traj_bytes).hexdigest()

trajectory_hashes = set()

for traj in trajectories:
    traj_hash = hash_trajectory(traj)
    trajectory_hashes.add(traj_hash)

unique_trajectories = len(trajectory_hashes)
print(f"Number of unique trajectories: {unique_trajectories} out of {total_trajectories}")
