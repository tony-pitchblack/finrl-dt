# let's read the test_loss_list.pkl file and train_loss_list.pkl file and plot them

import pickle
import matplotlib.pyplot as plt
import numpy as np  

path_prefix = "checkpoints/a2c_dt_lora_random_weight_gpt2_20742"

test_loss_list_file = f"{path_prefix}/total_asset_value_change_test_env_2000000.pkl"
train_loss_list_file = f"{path_prefix}/total_asset_value_change_train_env_4777057.587070737.pkl"

with open(test_loss_list_file, 'rb') as f:
    test_loss_list = pickle.load(f)

print(test_loss_list)

with open(train_loss_list_file, 'rb') as f:
    train_loss_list = pickle.load(f)

print(train_loss_list)
print(len(test_loss_list))
print(len(train_loss_list))

# plot the test_loss_list and train_loss_list
plt.plot(test_loss_list[:335], label='test loss')
plt.plot(train_loss_list[:2893], label='train loss')
plt.legend()
plt.show()
