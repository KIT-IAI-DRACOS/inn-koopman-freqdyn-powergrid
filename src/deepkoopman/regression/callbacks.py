"""Callbacks for monitoring Koopman operator parameters during training."""

import os
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.callbacks import Callback

import csv
import warnings

class KoopmanParameterMonitor(Callback):
    """
    Callback to record the evolution of Koopman operator parameters.
    
    Parameters
    ----------
    log_dir : str
        Directory to save logs and parameter plots
    save_every_n_epochs : int
        Save parameter states every N epochs
    visualize : bool
        Whether to generate visualization plots of parameter changes
    """
    def __init__(self, log_dir='koopman_params', save_every_n_epochs=5, visualize=True):
        super().__init__()
        self.log_dir = log_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.visualize = visualize
        
        # Store parameter history
        self.a_params_history = []
        self.b_params_history = []
        self.eigenvalues_history = []
        self.epochs = []
        
        # Store gradient information
        self.a_params_grad_history = []
        self.b_params_grad_history = []
        
        # Store parameter changes
        self.prev_a_params = None
        self.prev_b_params = None
        
        # Create log directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Adding state_dict and load_state_dict methods to make Lightning happy
    def state_dict(self):
        return {
            "a_params_history": self.a_params_history,
            "b_params_history": self.b_params_history,
            "eigenvalues_history": self.eigenvalues_history,
            "epochs": self.epochs
        }
    
    def load_state_dict(self, state_dict):
        self.a_params_history = state_dict["a_params_history"]
        self.b_params_history = state_dict["b_params_history"]
        self.eigenvalues_history = state_dict["eigenvalues_history"]
        self.epochs = state_dict["epochs"]
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Save current parameters at the start of each epoch for change calculation."""
        koopman_op = pl_module._koopman_propagator
        if self.prev_a_params is None:
            self.prev_a_params = koopman_op.a_params.clone().detach()
            self.prev_b_params = koopman_op.b_params.clone().detach()
        
    def on_after_backward(self, trainer, pl_module):
        """Record gradients after backward propagation."""
        koopman_op = pl_module._koopman_propagator
        
        # Record gradients of a_params and b_params
        if koopman_op.a_params.grad is not None:
            a_grad = koopman_op.a_params.grad.abs().mean().item()
            pl_module.log("koopman/a_params_grad", a_grad)
            
        if koopman_op.b_params.grad is not None:
            b_grad = koopman_op.b_params.grad.abs().mean().item()
            pl_module.log("koopman/b_params_grad", b_grad)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check parameter changes after each training batch."""
        if batch_idx % 50 == 0:  # Do not record every batch, too frequent
            koopman_op = pl_module._koopman_propagator
            # Compute parameter changes
            a_change = (koopman_op.a_params - self.prev_a_params).abs().mean().item()
            b_change = (koopman_op.b_params - self.prev_b_params).abs().mean().item()
            
            pl_module.log("koopman/a_params_change", a_change)
            pl_module.log("koopman/b_params_change", b_change)
            
            # Save current parameters
            self.prev_a_params = koopman_op.a_params.clone().detach()
            self.prev_b_params = koopman_op.b_params.clone().detach()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Record parameters at the end of each training epoch."""
        epoch = trainer.current_epoch
        
        # Get Koopman operator parameters
        koopman_op = pl_module._koopman_propagator
        # Record parameters
        a_params = koopman_op.a_params.detach().cpu().numpy().copy()
        b_params = koopman_op.b_params.detach().cpu().numpy().copy()
        
        # Compute eigenvalues
        eigenvalues = koopman_op.get_eigensystems()[0].detach().cpu().numpy().copy()
        eigenvalues = np.exp(eigenvalues*koopman_op.dt.detach().cpu().numpy())
        # Get gradients
        a_grad = koopman_op.a_params.grad
        b_grad = koopman_op.b_params.grad
        
        # Add to history
        self.a_params_history.append(a_params)
        self.b_params_history.append(b_params)
        self.eigenvalues_history.append(eigenvalues)
        self.epochs.append(epoch)
        
        # Record gradient information
        if a_grad is not None and b_grad is not None:
            self.a_params_grad_history.append(a_grad.abs().mean().item())
            self.b_params_grad_history.append(b_grad.abs().mean().item())
        
        # Log to logger
        for i, (a, b) in enumerate(zip(a_params, b_params)):
            pl_module.log(f"koopman/a_param_{i}", a, on_epoch=True)
            pl_module.log(f"koopman/b_param_{i}", b, on_epoch=True)
        
        # Log eigenvalue statistics
        eig_real = np.real(eigenvalues)
        eig_imag = np.imag(eigenvalues)
        eig_abs = np.abs(eigenvalues)
        
        pl_module.log("koopman/eigenvalue_max_abs", np.max(eig_abs), on_epoch=True)
        pl_module.log("koopman/eigenvalue_min_abs", np.min(eig_abs), on_epoch=True)
        pl_module.log("koopman/eigenvalue_mean_abs", np.mean(eig_abs), on_epoch=True)
        
        # Save and visualize every N epochs
        if epoch % self.save_every_n_epochs == 0 or epoch == trainer.max_epochs - 1:
            # self._save_parameters(epoch) # Temporarily not saving parameters to file
            if self.visualize:
                self._visualize_parameters(epoch)
    
    def _save_parameters(self, epoch):
        """Save parameter states to file."""
        save_path = os.path.join(self.log_dir, f"koopman_params_epoch_{epoch}.npz")
        np.savez(
            save_path,
            a_params=np.array(self.a_params_history),
            b_params=np.array(self.b_params_history),
            eigenvalues=np.array(self.eigenvalues_history),
            epochs=np.array(self.epochs),
            a_params_grad=np.array(self.a_params_grad_history) if len(self.a_params_grad_history) > 0 else np.array([0]),
            b_params_grad=np.array(self.b_params_grad_history) if len(self.b_params_grad_history) > 0 else np.array([0])
        )
    
    def _visualize_parameters(self, epoch):
        """Visualize parameter trends and eigenvalue distribution."""
        # Create figure directory
        fig_dir = os.path.join(self.log_dir, 'figures')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            
        # Create 2x2 layout, add gradient visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # a_params evolution
        a_params_array = np.array(self.a_params_history)
        for i in range(a_params_array.shape[1]):
            axes[0, 0].plot(self.epochs, a_params_array[:, i], label=f'a_{i}')
        axes[0, 0].set_title('figure of a_params')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Parameter Value')
        axes[0, 0].grid(True)
        # axes[0, 0].legend()
        
        # b_params evolution
        b_params_array = np.array(self.b_params_history)
        for i in range(b_params_array.shape[1]):
            axes[0, 1].plot(self.epochs, b_params_array[:, i], label=f'b_{i}')
        axes[0, 1].set_title('figure of b_params')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Parameter Value')
        axes[0, 1].grid(True)
        # axes[0, 1].legend()
        
        # Gradient evolution
        if len(self.a_params_grad_history) > 0:
            axes[1, 0].plot(self.epochs[:len(self.a_params_grad_history)], 
                          self.a_params_grad_history, label='a_params gradient')
            axes[1, 0].plot(self.epochs[:len(self.b_params_grad_history)], 
                          self.b_params_grad_history, label='b_params gradient')
            axes[1, 0].set_title('Parameter gradient variation')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Absolute value of the mean gradient')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
            axes[1, 0].legend()
        # Plot eigenvalue distribution on the unit circle
        current_eigenvalues = self.eigenvalues_history[-1]
        axes[1, 1].scatter(np.real(current_eigenvalues), np.imag(current_eigenvalues), 
                  c='blue', marker='o', alpha=0.7, label=f'Epoch {epoch}')
        
        # Plot unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
        axes[1, 1].add_patch(circle)
        
        # Set figure properties
        axes[1, 1].set_xlim(-1.5, 1.5)
        axes[1, 1].set_ylim(-1.5, 1.5)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True)
        axes[1, 1].set_title(f'eigenvalue distribution (Epoch {epoch})')
        axes[1, 1].set_xlabel('Real part')
        axes[1, 1].set_ylabel('imaginary part')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'params_history_epoch_{epoch}.png'))
        plt.close()


class LossRecorder(Callback):
    """
    Records loss values during training, saves them to a CSV file, and generates visualization plots.
    
    Parameters
    ----------
    log_dir : str, default='losses'
        Directory to save loss records
    csv_name : str, default='training_losses.csv'
        Name of the CSV file
    """
    def __init__(self, log_dir='losses', csv_name='training_losses.csv'):
        super().__init__()
        self.log_dir = log_dir
        self.csv_name = csv_name
        self.losses = []
    
    # Adding state_dict and load_state_dict methods to satisfy Lightning requirements
    def state_dict(self):
        return {"losses": self.losses}
    
    def load_state_dict(self, state_dict):
        self.losses = state_dict["losses"]
        
    def on_fit_start(self, trainer, pl_module):
        """Create log directory at the start of training."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create CSV file and write header
        self.csv_path = os.path.join(self.log_dir, self.csv_name)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'loss', 'rnn_loss', 'inv_loss'])
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record loss values at the end of each batch."""
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
            rnn_loss = outputs.get('rnn_loss', 0.0)
            inv_loss = outputs.get('inv_loss', 0.0)
        else:
            loss = outputs
            rnn_loss = getattr(pl_module, 'current_rnn_loss', 0.0)
            inv_loss = getattr(pl_module, 'current_inv_loss', 0.0)
        
        # Record to list
        self.losses.append({
            'epoch': trainer.current_epoch,
            'batch': batch_idx,
            'loss': loss.item() if hasattr(loss, 'item') else float(loss),
            'rnn_loss': rnn_loss.item() if hasattr(rnn_loss, 'item') else float(rnn_loss),
            'inv_loss': inv_loss.item() if hasattr(inv_loss, 'item') else float(inv_loss)
        })
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trainer.current_epoch, 
                batch_idx,
                self.losses[-1]['loss'],
                self.losses[-1]['rnn_loss'],
                self.losses[-1]['inv_loss']
            ])
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Calculate average losses at the end of each epoch."""
        epoch_losses = [l for l in self.losses if l['epoch'] == trainer.current_epoch]
        if epoch_losses:
            avg_loss = sum(l['loss'] for l in epoch_losses) / len(epoch_losses)
            avg_rnn_loss = sum(l['rnn_loss'] for l in epoch_losses) / len(epoch_losses)
            avg_inv_loss = sum(l['inv_loss'] for l in epoch_losses) / len(epoch_losses)
            
            # Record epoch averages to CSV
            epoch_csv_path = os.path.join(self.log_dir, 'epoch_losses.csv')
            write_header = not os.path.exists(epoch_csv_path) or os.path.getsize(epoch_csv_path) == 0
            
            with open(epoch_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:  # Write header if writing for the first time
                    writer.writerow(['epoch', 'avg_loss', 'avg_rnn_loss', 'avg_inv_loss'])
                writer.writerow([
                    trainer.current_epoch,
                    avg_loss,
                    avg_rnn_loss,
                    avg_inv_loss
                ])
                
            # Plot and save loss curves
            self.plot_losses(trainer.current_epoch)
    
    def plot_losses(self, current_epoch):
        """Plot and save loss curves."""
        try:
            # Extract data
            epochs = [l['epoch'] for l in self.losses]
            batch_indices = [l['batch'] for l in self.losses]
            global_steps = [e * 100 + b for e, b in zip(epochs, batch_indices)]  # Estimated global step
            losses = [l['loss'] for l in self.losses]
            rnn_losses = [l['rnn_loss'] for l in self.losses]
            inv_losses = [l['inv_loss'] for l in self.losses]
            
            # Create plots
            plt.figure(figsize=(15, 10))
            
            # Total loss curve
            plt.subplot(2, 2, 1)
            plt.plot(global_steps, losses, 'b-', alpha=0.6, label='Total Loss')
            plt.title('Training Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # RNN loss curve
            plt.subplot(2, 2, 2)
            plt.plot(global_steps, rnn_losses, 'r-', alpha=0.6, label='RNN Loss')
            plt.title('RNN Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # curve
            # plt.subplot(2, 2, 3)
            # plt.plot(, , 'g-', alpha=0.6, label='')
            # plt.title('')
            # plt.xlabel('Steps')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.grid(True, alpha=0.3)
            
            # Invertibility loss curve
            plt.subplot(2, 2, 4)
            plt.plot(global_steps, inv_losses, 'm-', alpha=0.6, label='Invertibility Loss')
            plt.title('Invertibility Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plots
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f'loss_plot_epoch_{current_epoch}.png'))
            
            # Log-scale plots
            plt.figure(figsize=(15, 10))
            
            # Total loss curve (log scale)
            plt.subplot(2, 2, 1)
            plt.semilogy(global_steps, losses, 'b-', alpha=0.6, label='Total Loss')
            plt.title('Training Loss (log scale)')
            plt.xlabel('Steps')
            plt.ylabel('Loss (log)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # RNN loss curve (log scale)
            plt.subplot(2, 2, 2)
            plt.semilogy(global_steps, rnn_losses, 'r-', alpha=0.6, label='RNN Loss')
            plt.title('RNN Loss (log scale)')
            plt.xlabel('Steps')
            plt.ylabel('Loss (log)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save log-scale plots
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f'loss_plot_log_epoch_{current_epoch}.png'))
            plt.close('all')
            
            # Plot average loss per epoch
            epoch_nums = []
            epoch_avg_losses = []
            epoch_avg_rnn_losses = []
            
            for ep in range(current_epoch + 1):
                ep_losses = [l['loss'] for l in self.losses if l['epoch'] == ep]
                ep_rnn_losses = [l['rnn_loss'] for l in self.losses if l['epoch'] == ep]
                if ep_losses:
                    epoch_nums.append(ep)
                    epoch_avg_losses.append(sum(ep_losses) / len(ep_losses))
                    epoch_avg_rnn_losses.append(sum(ep_rnn_losses) / len(ep_rnn_losses))
            
            plt.figure(figsize=(12, 6))
            plt.semilogy(epoch_nums, epoch_avg_losses, 'bo-', label='Avg Total Loss')
            plt.semilogy(epoch_nums, epoch_avg_rnn_losses, 'ro-', label='Avg RNN Loss')
            plt.title('Average Losses per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss (log)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.log_dir, f'epoch_avg_losses.png'))
            plt.close()
            
        except Exception as e:
            warnings.warn(f"Unable to plot loss curves: {str(e)}")
