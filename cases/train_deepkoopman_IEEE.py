import os
import sys
import numpy as np
from pathlib import Path
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
src_path = os.path.join(parent_dir, "src")
print("Füge zu sys.path hinzu:", src_path)
sys.path.append(src_path)
sys.path.append(parent_dir)
sys.path.append("pytorch-i-revnet")

from data.data_process.data_loader_multi_wecc import prepare_data
from deepkoopman.regression import DeepKoopman

# ========== Load Data ==========
fault_idx = [1,2,3,4,5,6,7,8,9,11,14]
gen_buses = [1,2,3,6,8]
data_folder_path = os.path.abspath(os.path.join(base_dir, "..", "data/data_folder/IEEE14_1"))
data = prepare_data(data_folder_path, fault_idx, 'ieee14_fault_1.1', 1.1, phasor_buses=gen_buses)

omega_list =  data['omega_list']
delta_list =  data['delta_list']
time_list =   data['time_list']

N_traj = len(fault_idx)
np.random.seed(0)
all_indices = np.arange(N_traj)
np.random.shuffle(all_indices)

n_train = 9
train_indices = all_indices[:n_train].tolist()
test_indices  = all_indices[n_train:].tolist()

train_x = [np.concatenate(((omega_list[ii]-1)*2*np.pi*60, delta_list[ii]), axis=1) for ii in train_indices]
test_x  = [np.concatenate(((omega_list[ii]-1)*2*np.pi*60, delta_list[ii]), axis=1) for ii in test_indices]
train_time = [time_list[ii] for ii in train_indices]
test_time = [time_list[ii] for ii in test_indices]

print("Training trajectories:", len(train_x))
print("Test trajectories:", len(test_x))
print("Sample shape:", train_x[0].shape)

# ========== Model Setup ==========
data_dt = data['time_list'][0][1] - data['time_list'][0][0]
state_dim = 10
time_delay_embed = 4
extension_dim = 0
inn_type = "iresnet"
extension_type = "cnn"

dlk_regressor = DeepKoopman(
    dt=data_dt,
    look_forward=8,
    time_delay=time_delay_embed,
    config_inn=dict(
        input_size=state_dim * time_delay_embed,
        hidden_size=64 * time_delay_embed,
        num_layers=4,
        extension_config=None,
        output_size=state_dim * time_delay_embed + extension_dim,
        init_identity=True,
        inn_type=inn_type,
    ),  
    batch_size=64,
    normalize=True,
    progressive_steps=True,
    trainer_kwargs={
        'max_epochs': 50,
        'accelerator': 'cpu',  # oder 'gpu' wenn verfügbar
        'gradient_clip_val': 0.2,
        'log_every_n_steps': 5
    },
)


# ========== Train Model ==========
os.makedirs('test_'+inn_type, exist_ok=True)

dlk_regressor.fit(
    x=train_x,
    y=None,
    monitor_params=True,
    log_dir='test_'+inn_type,
    record_losses=True
)




"""
# ========== Save Model ==========
save_dict = {
    "inn_state_dict": dlk_regressor._regressor.state_dict(),
    "inn_type": inn_type,
    "config_inn": dlk_regressor.config_inn,
    "look_forward": dlk_regressor.look_forward,
    "time_delay": dlk_regressor.time_delay,
    "losses": getattr(dlk_regressor, "losses_", None),
    "eigenvalues": dlk_regressor._eigenvalues_,
    "eigenvectors": dlk_regressor._eigenvectors_,
    "koopman_operator": dlk_regressor._state_matrix_
}



save_path = f"deephodmd_results"+inn_type+"/dlk_{inn_type}_{extension_type}.pth"
torch.save(save_dict, save_path)
print(f"✅ Modell erfolgreich gespeichert unter: {save_path}")

"""