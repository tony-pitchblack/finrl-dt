import pickle

with open('data/train_a2c_trajectory_2024-10-13_12-47-12.pkl', 'rb') as f:
    data = pickle.load(f)

data = data[1]
print(data)
print(len(data['actions']))
print(len(data['rewards']))
