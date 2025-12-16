import os
import sys
import numpy as np
from pathlib import Path
import torch
import wandb
import ast
from sklearn.metrics import mean_squared_error as mse

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
sys.path.append(parent_dir)
sys.path.append("pytorch-i-revnet")

from data.data_process.data_loader_multi_wecc import prepare_data
from deepkoopman.regression import DeepKoopman

def train():
    # === WANDB init ===
    wandb.init(
        project="koopman-inn-comparison",
        name=f"train-{os.getenv('WANDB_RUN_NAME', 'run')}"
    )
    config = wandb.config

    # === Load Data ===
    fault_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14]
    gen_buses = [1, 2, 3, 6, 8]
    data_folder_path = os.path.abspath(os.path.join(base_dir, "..", "data/data_folder/IEEE14_1"))
    data = prepare_data(data_folder_path, fault_idx, 'ieee14_fault_1.1', 1.1, phasor_buses=gen_buses)

    omega_list = data['omega_list']
    delta_list = data['delta_list']
    time_list = data['time_list']

    N_traj = len(fault_idx)
    np.random.seed(0)
    all_indices = np.arange(N_traj)
    np.random.shuffle(all_indices)

    N_traj = len(fault_idx)
    np.random.seed(0)
    all_indices = np.arange(N_traj)
    np.random.shuffle(all_indices)

    n_train = 7
    n_val = 2
    n_test = N_traj - n_train - n_val

    train_indices = all_indices[:n_train].tolist()
    val_indices = all_indices[n_train:n_train + n_val].tolist()
    test_indices = all_indices[n_train + n_val:].tolist()

    train_x = [np.concatenate(((omega_list[ii]-1)*2*np.pi*60, delta_list[ii]), axis=1) for ii in train_indices]
    val_x = [np.concatenate(((omega_list[ii]-1)*2*np.pi*60, delta_list[ii]), axis=1) for ii in val_indices]
    test_x = [np.concatenate(((omega_list[ii]-1)*2*np.pi*60, delta_list[ii]), axis=1) for ii in test_indices]



    train_x = [np.concatenate(((omega_list[ii]-1)*2*np.pi*60, delta_list[ii]), axis=1) for ii in train_indices]
    test_x = [np.concatenate(((omega_list[ii]-1)*2*np.pi*60, delta_list[ii]), axis=1) for ii in test_indices]
    state_dim = 10 # always 10 because data is shape 10 

    # === Dynamische Ableitungen ===
    input_size = state_dim * config.time_delay
    output_size = input_size #+ config.extension_output_size  # falls du das brauchst

    """
    # Konvertiere Strings zu Listen (wenn nötig)
    try:
        hidden_channels = ast.literal_eval(config.hidden_channels) if isinstance(config.hidden_channels, str) else config.hidden_channels
        kernel_sizes = ast.literal_eval(config.kernel_sizes) if isinstance(config.kernel_sizes, str) else config.kernel_sizes
    except Exception as e:
        raise ValueError(f"Fehler beim Parsen von hidden_channels oder kernel_sizes: {e}")
    """
    # === Modell definieren ===
    data_dt = data['time_list'][0][1] - data['time_list'][0][0]

    """
    if config.use_extension:
        extension_config = dict(
            extension_type="cnn",
            extension_output_size=config.extension_output_size,
            hidden_channels=config.hidden_channels,
            kernel_sizes=config.kernel_sizes,
            dropout_rate=config.dropout_rate,
            use_residual=config.use_residual
        )
        output_size = input_size + config.extension_output_size
    else:
        extension_config = None
        output_size = input_size
    """

    dlk_regressor = DeepKoopman(
        dt=data_dt,
        look_forward=config.look_forward,
        time_delay=config.time_delay,
        config_inn=dict(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=output_size,
            init_identity=True,
            inn_type="freiaallinone",
        ),
        extension_config=None,
        batch_size=64,
        normalize=config.normalize,
        progressive_steps=True,
        trainer_kwargs={
            'max_epochs': 30,
            'accelerator': "cpu",
            'gradient_clip_val': 0.2,
            'log_every_n_steps': 5
        },
    )

    os.makedirs('deephodmd_results', exist_ok=True)
    dlk_regressor.fit(
        x=train_x,
        y=None,
        monitor_params=True,
        log_dir='deephodmd_results',
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


if __name__ == "__main__":
    train()


