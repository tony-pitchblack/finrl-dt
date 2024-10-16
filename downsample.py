import os
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime

def downsample_trajectories(input_pickle, output_dir, sample_ratios, random_seed=42):
    """
    Downsamples trajectories from a pickle file based on specified sample ratios.

    Args:
        input_pickle (str): Path to the input pickle file containing trajectories.
        output_dir (str): Directory where downsampled pickle files will be saved.
        sample_ratios (list of float): List of sample ratios (e.g., [0.01, 0.05, 0.1]).
        random_seed (int): Seed for random number generator to ensure reproducibility.

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Load the original trajectories
    with open(input_pickle, 'rb') as f:
        trajectories = pickle.load(f)

    num_trajectories = len(trajectories)
    print(f"Total number of trajectories in the original pickle: {num_trajectories}")

    for ratio in sample_ratios:
        if ratio <= 0 or ratio > 1:
            print(f"Invalid sample ratio {ratio}. It should be between 0 and 1.")
            continue

        sampled_trajectories = []

        if num_trajectories > 1:
            # **Multiple Trajectories:** Sample entire trajectories
            num_sample = max(1, int(num_trajectories * ratio))
            sampled_indices = np.random.choice(num_trajectories, size=num_sample, replace=False)
            sampled_trajectories = [trajectories[i] for i in sampled_indices]
            print(f"Sample Ratio: {ratio*100:.2f}% | Sampled Trajectories: {num_sample}/{num_trajectories}")
        elif num_trajectories == 1:
            # **Single Trajectory:** Sample individual steps
            trajectory = trajectories[0]
            total_steps = len(trajectory['actions'])  # Assuming all arrays are of the same length

            num_sample_steps = max(1, int(total_steps * ratio))
            sampled_step_indices = np.sort(np.random.choice(total_steps, size=num_sample_steps, replace=False))

            sampled_observations = trajectory['observations'][sampled_step_indices]
            sampled_actions = trajectory['actions'][sampled_step_indices]
            sampled_rewards = trajectory['rewards'][sampled_step_indices]
            sampled_dones = trajectory['terminals'][sampled_step_indices]

            # Reconstruct the trajectory with sampled steps
            sampled_trajectory = {
                'observations': sampled_observations,
                'actions': sampled_actions,
                'rewards': sampled_rewards,
                'terminals': sampled_dones
            }
            sampled_trajectories.append(sampled_trajectory)
            print(f"Sample Ratio: {ratio*100:.2f}% | Sampled Steps: {num_sample_steps}/{total_steps}")
        else:
            print("No trajectories found in the input pickle.")
            continue

        # Define the output filename
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if num_trajectories > 1:
            output_filename = f'downsampled_trajectories_{ratio*100:.0f}pct_{current_time}.pkl'
        else:
            output_filename = f'downsampled_steps_{ratio*100:.0f}pct_{current_time}.pkl'
        
        output_path = os.path.join(output_dir, output_filename)

        # Save the downsampled trajectories to a new pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(sampled_trajectories, f)

        print(f"Downsampled data saved to '{output_path}'\n")

if __name__ == "__main__":
    # Example Usage
    input_pickle = 'trajectories_a2c_1_2024-10-06_14-19-28.pkl'  # Replace with your actual pickle file
    output_dir = 'downsampled_trajectories'  # Directory to save downsampled pickle files
    sample_ratios = [0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]  # Define your desired sample ratios (as fractions)

    downsample_trajectories(input_pickle, output_dir, sample_ratios)
