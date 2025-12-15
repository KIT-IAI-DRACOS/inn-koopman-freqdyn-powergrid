import os, sys
from pathlib import Path
import numpy as np
import warnings
import torch
import sys
import wandb
from sklearn.metrics import mean_squared_error as mse


base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
src_path = os.path.join(parent_dir, "src")
print("Füge zu sys.path hinzu:", src_path)
sys.path.append(src_path)
sys.path.append(parent_dir)

from pykoopman.regression import DeepHODMD
from data.data_process.data_loader_multi_wecc import prepare_data
from utils.add_noise import add_noise_by_snr_numpy

wandb.init(
    project="WECC-inn-comparison",
    name=f"train-{os.getenv('WANDB_RUN_NAME', 'run')}"
)
config = wandb.config
"""
fault_idx = [1, 2, 3, 8, 9, 12, 14, 16, 17, 21, 25, 26, 27, 29, 30, 31, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 66, 67, 70, 72, 73, 74, 83, 98, 99, 100, 101, 102, 103, 104, 105,
            106, 108, 109, 111, 112, 115, 116, 117, 123, 125, 128, 133, 135, 136, 137,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155,
            156, 157, 159, 160, 162, 163, 164, 165, 166]
"""
fault_idx = [3, 8, 9, 12, 14, 17, 21, 25, 26, 27, 29, 30, 31, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58, 59, 60,
            62, 66, 67, 70, 73, 74, 83, 98, 99, 102, 103, 104, 105,
            106, 108, 109, 111, 112, 115, 116, 117, 123, 125, 128, 133, 135, 136, 137,
            141, 142, 143, 145, 146, 147, 148, 151, 153, 155,
            156, 159, 160, 162, 164, 165]
gen_buses = [3, 5, 8, 10, 12, 14, 17, 29, 34, 35, 39, 42, 44, 46, 64, 69, 76, 78, 102, 111, 115,
             117, 137, 139, 143, 147, 148, 158, 161]
N_traj = len(fault_idx)
data_folder_path = os.path.abspath(os.path.join(base_dir, ".."))+os.sep+'data/data_folder/WECC_1'

# data12 = prepare_data(data_folder_path, fault_idx, 'ieee14_fault_1.2', 1.2)
data = prepare_data(data_folder_path, fault_idx, 'wecc_full_20', 1.1, phasor_buses = gen_buses)

omega_list =  data['omega_list']
delta_list =  data['delta_list']
time_list =  data['time_list']

# Dataset splitting
np.random.seed(0)  # Set random seed for reproducibility
all_indices = np.arange(N_traj)
np.random.shuffle(all_indices)

# Split the dataset
n_train = 60  # Training set size
n_val = 10

# Get indices for training and testing sets
train_indices = all_indices[:n_train].tolist()
val_indices   = all_indices[n_train:n_train + n_val].tolist()
test_indices  = all_indices[n_train + n_val:].tolist()
train_Ntraj = len(train_indices)
val_Ntraj   = len(val_indices)
test_Ntraj  = len(test_indices)

# Create input data lists
# Reshape data to [n_timesteps, n_features] * n_trajectories
train_x = [np.concatenate((
    (omega_list[ii]-1)*2*np.pi*60,
    delta_list[ii],
    # data['pe_list'][ii],
    # data['voltage_list'][ii]
), axis=1) for ii in train_indices]          
n_vars_x = train_x[0].shape[-1]                   # = omega + delta + pe

val_x = [np.concatenate((
    (omega_list[ii]-1)*2*np.pi*60,
    delta_list[ii],
    # data['pe_list'][ii],
    # data['voltage_list'][ii]
), axis=1) for ii in val_indices]  

# Reshape data to  [n_timesteps, n_features] * n_trajectories 
test_x = [np.concatenate((
    (omega_list[ii]-1)*2*np.pi*60,
    delta_list[ii],
    # data['pe_list'][ii],
    # data['voltage_list'][ii]
), axis=1) for ii in test_indices]          # Shape: [n_timesteps, n_features] * n_trajectories

# Print dataset split information
print(f"Total trajectories: {N_traj}")
print(f"Training trajectories per fold: {len(train_indices)}")
# print(f"Test trajectories: {len(test_indices)}")

# Process time data similarly
train_time = [time_list[ii] for ii in train_indices]
val_time = [time_list[ii] for ii in val_indices]
test_time = [time_list[ii] for ii in test_indices]

# [Add noise]
train_x_noise = [None]*(len(train_x))
for ii in range(len(train_x)):
    train_x_noise[ii] = add_noise_by_snr_numpy(train_x[ii], 40)
    train_x_noise[ii] = train_x_noise[ii]

val_x_noise = [None]*(len(val_x))
for ii in range(len(val_x)):
    val_x_noise[ii] = add_noise_by_snr_numpy(val_x[ii], 40)
    val_x_noise[ii] = val_x_noise[ii]

test_x_noise = [None]*(len(test_x))
for ii in range(len(test_x)):
    test_x_noise[ii] = add_noise_by_snr_numpy(test_x[ii], 40)
    test_x_noise[ii] = test_x_noise[ii]

# Print data shapes for verification
print("Training data shape:", train_x[0].shape)  # [n_timesteps, n_features] * n_train_trajectories
# print("Test data shape:", test_x[0].shape)   # [n_timesteps, n_features] * n_test_trajectories

print("Number of training trajectories:", len(train_x))
# print("Number of test trajectories:", len(test_x))



torch.cuda.empty_cache()



sys.path.append('../src')

# import numpy.random as rnd
np.random.seed(42)  # for reproducibility

# Explicitly disable CUDA in PyTorch
import torch
torch.set_float32_matmul_precision('high')
# torch.cuda.is_available = lambda: False  # Force PyTorch to think CUDA is not available
# device = torch.device('cpu')  # Explicitly set device to CPU



warnings.filterwarnings('ignore')


#look_forward = 128
data_dt = data['time_list'][0][1]-data['time_list'][0][0]
state_dim = 58
time_delay_embed = config.time_delay
extension_dim = 0
inn_type = "freiaallinone"


# Create the two-stage model
dlk_regressor = DeepHODMD(
    dt=data_dt,
    look_forward=config.look_forward,
    time_delay=time_delay_embed,
    config_inn=dict(
        input_size=state_dim * time_delay_embed,
        hidden_size=config.hidden_size * time_delay_embed,
        num_layers=2,
        extension_config=None, 
        output_size=state_dim * time_delay_embed,
        init_identity=True,
        inn_type = inn_type,
    ),
    
    batch_size=128,
    normalize=True,
    # extension_hidden_size=32,       # Hidden size of the extension layer
    # extension_num_layers=3,         # Number of layers in the extension layer
    progressive_steps=True,         # Whether to use progressive prediction steps
    trainer_kwargs={'max_epochs': 2, 
                    'gradient_clip_val': 0.1,
                    'accelerator': 'cpu'},
)

os.makedirs('WECC_results', exist_ok=True)
# Train with the two-stage approach
dlk_regressor.fit(
    x=train_x,
    y=None,
    monitor_params=True,
    log_dir='WECC_results',
    record_losses=True
)


errors = []

for true_traj in val_x:  # val_x = list of arrays
    x0 = true_traj[:config.time_delay].reshape(1, -1)  # Startzustand für delay
    true = true_traj[config.time_delay:]               # "wahre" Zukunft
    
    pred_traj = dlk_regressor.simulate(x0, n_steps=len(true))  # ganze vorhergesagte Trajektorie
    pred = pred_traj[1:]                                 # ohne Startzustand
    
    # Optional: nur state_dim extrahieren
    true = true[:, :state_dim]
    pred = pred[:, :state_dim]

    assert true.shape == pred.shape
    errors.append(mse(true, pred))



val_loss = np.mean(errors)
wandb.log({"val_loss": val_loss})




wandb.finish()