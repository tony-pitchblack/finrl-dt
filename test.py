import pickle
with open('data/test_a2c_trajectory_2024-10-12_14-37-42.pkl', "rb") as f:
    train_trajectory = pickle.load(f)
env_targets_train = [train_trajectory[0]['rewards'].sum() / 1000] # this is the total return of the training trajectory
