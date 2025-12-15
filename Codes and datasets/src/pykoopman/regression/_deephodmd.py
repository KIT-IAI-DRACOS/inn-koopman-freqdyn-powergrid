"""实现新的Koopman学习方法"""
import torch
import torch.nn.functional as F
import numpy as np
import lightning as L
import os
import torcheval.metrics.functional as FM
from typing import Dict, Any, Optional, Union, List, Tuple

from pykoopman.regression.inns.freia_allinone import FreiaINN
from pykoopman.regression._extensionlayer import CNNExtensionLayer
from pykoopman.regression._koopmanoperator import StableKoopmanOperator,  OrthogonalLinear

from .callbacks import KoopmanParameterMonitor, LossRecorder
from sklearn.utils.validation import check_is_fitted
from pykoopman.regression._base import BaseRegressor
from pykoopman.regression._nndmd import SeqDataModule

from pykoopman.regression.inns._inn_factory import INNFactory
from pykoopman.regression._extension_factory import ExtensionFactory
from pykoopman.regression.inns._hybrid_inn_wrapper import HybridINNWrapper
from torcheval.metrics.functional import r2_score


class DeepKoopmanRegressor(L.LightningModule):
    """
    New Koopman Regressor

    Parameters
    ----------
    dt : float, default=1.0
        Time step
    look_forward : int, default=1
        Prediction steps
    config_inn : dict
        Invertible neural network configuration
    extension_hidden_size : int, default=128
        Hidden size of the extension layer
    extension_num_layers : int, default=5
        Number of layers in the extension layer
    progressive_steps : bool, default=True
        Whether to use progressive training strategy
    """
    def __init__(
        self,
        dt: float = 1.0,
        look_forward: int = 1,
        config_inn: Dict[str, Any] = {},
        extension_config: Optional[Dict[str, Any]] = None,
        progressive_steps: bool = True,
    ):
        super().__init__()
        
        #self.config_inn = config_inn
        self.input_size = config_inn["input_size"]
        self.output_size = config_inn["output_size"]
        
        
        # Create the base INN
        inn = INNFactory.create_inn(config_inn)
        self.input_size = inn.input_size
        base_output_size = inn.output_size



        # Calculate the required output dimension for the extension
        self.extension_type = None
        extension_output_size = 0

        # Only process if extension config is provided
        if extension_config is not None:
            self.extension_type = extension_config.get("extension_type", None)
            extension_output_size = extension_config.get("extension_output_size", 0)

        if self.extension_type is not None and extension_output_size > 0:
            if extension_config is None:
                extension_config = {
                    "hidden_size": 128,
                    "num_layers": 5,
                }

            extension_net = ExtensionFactory.create_extension(
                name=self.extension_type,
                input_size=self.input_size,
                output_size=extension_output_size,
                config=extension_config,
            )


            self._inn = HybridINNWrapper(base_inn=inn, extension_net=extension_net)
            self._extension = extension_net

        else:
            self._inn = inn
            self._extension = None

        self.feature_size = base_output_size + extension_output_size
        
        # Invertible linear mapping layer w(f(x), g(x))
        if extension_output_size > 0:
            self._linear_layer = OrthogonalLinear(
                self._inn.output_size, 
                init_mode='swap_blocks',
                block1_size=self.input_size
            )
        else:
            self._linear_layer = OrthogonalLinear(self.output_size)
        

        # In DeepKoopmanRegressor.__init__
        self._inn = INNFactory.create_inn(config_inn)
        if self._inn.input_size < self._inn.output_size:
            self._linear_layer = OrthogonalLinear(self._inn.output_size, init_mode="swap_blocks", block1_size=self._inn.input_size)
        else:
            self._linear_layer = OrthogonalLinear(self._inn.output_size)

        self.feature_size = self._inn.output_size

        

        # Dimensionality adjustment for Koopman Matrix
        # Koopman operator
        self._koopman_propagator = StableKoopmanOperator(
            dim=self.feature_size,
            dt=dt,
            # bandwidth=5,
        )


        # Config parameters
        self.look_forward = look_forward
        self.progressive_steps = progressive_steps
        # self.current_epoch = 0
        
        # Progressive training parameters
        if progressive_steps:
            self.min_look_ahead = 1
            self.max_look_ahead = look_forward
            self.current_look_ahead = self.min_look_ahead
        else:
            self.current_look_ahead = look_forward
        
        # Save hyperparameters
        self.save_hyperparameters()

    def _ensure_on_device(self, tensor):
        """Ensure tensor is on current device"""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor
    
    def _process_2d(self, x, rev=False):
        if not rev:
            # Encode with HybridINNWrapper or BaseINN
            phi = self._inn.forward(x)

            # Falls eine Extension existiert (nur bei HybridINNWrapper relevant), Logik ins Model ausgelagert
            if hasattr(self._inn, "get_extension_contribution") and self.training:
                contrib_ratio = self._inn.get_extension_contribution()
                self.log("ext_contrib_ratio", contrib_ratio, prog_bar=True)

            # Apply linear Koopman mapping
            return self._linear_layer(phi)

        else:
            # Inverse Koopman mapping
            decoded_combined = self._linear_layer.inverse(x)

            # Decode with HybridINNWrapper or BaseINN
            x_recon = self._inn.inverse(decoded_combined)
            return x_recon

    def _encode(self, x):
        """
        Encoding process: x -> phi
        Supports 2D tensors (batch_size or sequence_length, input_size) and 3D tensors (batch_size, sequence_length, input_size)
        """
        # Check input dimension
        if x.dim() == 2:
            # 2D input (others, input_size)
            return self._process_2d(x)
        elif x.dim() == 3:
            
            # 3D input (batch_size, sequence_length, input_size)
            batch_size, seq_len, features = x.shape
            
            # Check if feature dimension matches
            if features != self.input_size:
                raise ValueError(f"Input feature dimension {features} doesn't match expected {self.input_size}")
            
            # Reshape to 2D tensor (batch_size*seq_len, input_size)
            x_reshaped = x.reshape(-1, features)

            # Process 2D tensor
            phi_2d = self._process_2d(x_reshaped)
            
            # Reshape back to 3D tensor (batch_size, seq_len, output_size)
            phi_features = phi_2d.shape[1]
            phi_3d = phi_2d.reshape(batch_size, seq_len, phi_features)
            
            return phi_3d
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, only 2D or 3D tensors are supported")
    
    def _decode(self, phi):
        """
        Decoding process: phi -> x'
        Supports 2D and 3D tensors
        """
        # Check input dimension
        if phi.dim() == 2:
            # 2D input
            return self._process_2d(phi, rev=True)
        elif phi.dim() == 3:
            # 3D input
            batch_size, seq_len, features = phi.shape
            
            # Reshape to 2D tensor
            phi_reshaped = phi.reshape(-1, features)
            
            # Process 2D tensor
            x_recon_2d = self._process_2d(phi_reshaped, rev=True)
            
            # Reshape back to 3D tensor
            x_features = x_recon_2d.shape[1]
            x_recon_3d = x_recon_2d.reshape(batch_size, seq_len, x_features)
            
            return x_recon_3d
        else:
            raise ValueError(f"Unsupported input dimension: {phi.dim()}, only 2D or 3D tensors are supported")

    @torch.no_grad()
    def forward(self, x, n=1, record=True):
        """
        Forward propagation, predict n steps

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        n : int, default=1
            Number of prediction steps
        record : bool, default=True
            Whether to record intermediate states
            
        Returns
        -------
        torch.Tensor
            Prediction result
        """
        try:
            x = self._ensure_on_device(x)
            phi = self._encode(x)
            phi_seq = self._koopman_propagator.forward(phi, n, record=record)
            decoded = self._decode(phi_seq)
            return decoded
            
        except Exception as e:
            print(f"Forward propagation error: {str(e)}")
            raise

    def orthogonality_regularization(f_outputs, g_outputs = None):
        """
        Compute orthogonality regularization between f and g outputs

        Parameters
        ----------
        f_outputs : torch.Tensor
            Output of invertible network f, shape (batch_size, dim_f)
        g_outputs : torch.Tensor
            Output of extension layer g, shape (batch_size, dim_g)
            
        Returns
        -------
        torch.Tensor
            Orthogonality regularization loss
        """
        
        orth_loss = 0.0

        if g_outputs is not None:
            # Normalize outputs
            f_normalized = F.normalize(f_outputs, dim=1)
            g_normalized = F.normalize(g_outputs, dim=1)

            # Compute cosine similarity matrix
            similarity = torch.mm(f_normalized, g_normalized.t())

            # Compute orthogonality loss as squared Frobenius norm
            orth_loss = torch.sum(similarity ** 2) / (similarity.shape[0] * similarity.shape[1])

            # Compute cosine similarity matrix for g
            similarity = torch.mm(g_normalized, g_normalized.t())

            orth_loss += 0.5* torch.sum(similarity ** 2) / (similarity.shape[0] * similarity.shape[1])            
        
        return orth_loss

    def _compute_loss(self, batch):
        """
        Compute training loss

        Parameters
        ----------
        batch : tuple
            Batch data containing (x, y, ys)
        
        Returns
        -------
        tuple
            Tuple containing different loss values
        """
        # Unpack data
        x, y, ys = batch
        x = self._ensure_on_device(x)
        y = self._ensure_on_device(y)
        ys = self._ensure_on_device(ys)
        
        # Get max prediction steps in batch
        batch_look_forward = int(ys.max().item()) if ys.numel() > 0 else 1
        batch_look_forward = min(batch_look_forward, self.current_look_ahead)



        # Default loss values
        rnn_loss = torch.tensor(0.0, device=self.device)
        inv_loss = torch.tensor(0.0, device=self.device)


       
        
        # Forward propagation and loss calculation
        try:
            # Encode input
            phi_x = self._encode(x)
            # Predict future states using Koopman operator
            if phi_x.dim() == 2:
                # (batch_size, n_features) -> (batch_size, seq_len, n_features)
                phi_seq = self._koopman_propagator.forward(phi_x.unsqueeze(1), n=batch_look_forward, record=True)
            else:
                phi_seq = self._koopman_propagator.forward(phi_x, n=batch_look_forward, record=True)
            
            # Decode prediction sequence
            if batch_look_forward>0:
                decoded_y_seq = self._decode(phi_seq)
                if batch_look_forward < self.look_forward:
                    padding = torch.zeros(
                        (x.size(0), self.look_forward - batch_look_forward, self.input_size),
                        device=self.device
                    )
                    decoded_y_seq = torch.cat([decoded_y_seq, padding], dim=1)
                    
                    padding = torch.zeros(
                        (phi_seq.size(0), self.look_forward - batch_look_forward, self.feature_size),
                        device=self.device
                    )
                    phi_seq = torch.cat([phi_seq, padding], dim=1)
                else:
                    decoded_y_seq = decoded_y_seq[:, :self.look_forward, :]
                    phi_seq = phi_seq[:, :self.look_forward, :]
            else:
                decoded_y_seq = torch.zeros(
                    (x.size(0), self.look_forward, self.input_size), 
                    device=self.device
                )
                phi_seq = torch.zeros(
                    (phi_seq.size(0), self.look_forward, self.output_size), 
                    device=self.device
                )

            # Compute prediction loss (time domain)
            rnn_loss = self._masked_mse_loss(decoded_y_seq, y, ys)

            
            batch_size, seq_len, n_features = decoded_y_seq.shape
            decoded_flat = decoded_y_seq.reshape(batch_size*seq_len, n_features)
            y_flat = y.reshape(batch_size*seq_len, n_features)
            

            koopman_loss = self._masked_mse_loss(phi_seq, self._encode(y), ys)

            phi_x_inverse = torch.zeros_like(
                phi_x,
                device=self.device
            )

            # Backward reconstruction loss (time domain)
            if batch_look_forward > 0:
                if y.dim() == 2:
                    # (batch_size, n_features)
                    phi_x_inverse = self._koopman_propagator.inverse(self._encode(y), n = 1, record = False)
                else:
                    # (batch_size, seq_len, n_features)
                    # Use the last time step of phi_seq directly
                    phi_x_inverse = self._koopman_propagator.inverse(self._encode(y[:,batch_look_forward-1,:]), n = batch_look_forward, record = False)
            
            # x_inverse = self._decode(phi_x_inverse)
            # koopman_loss = F.smooth_l1_loss(phi_x_inverse, phi_x, beta=5e-3)
            koopman_loss = F.mse_loss(phi_x_inverse, phi_x)

            # Compute invertibility loss
            # Reconstruct input x
            decoded_x = self._decode(phi_x)
            inv_loss = F.mse_loss(decoded_x, x)
            
            # Encode and reconstruct target sequence y
            if y is not None and y.numel() > 0:
                 for i in range(min(batch_look_forward, y.shape[1])):
                     y_i = y[:, i, :]
                     # Skip all-zero or invalid samples
                     if torch.all(y_i == 0):
                         continue
                     phi_y = self._encode(y_i)
                     reconstructed_y = self._decode(phi_y)
                     inv_loss += F.mse_loss(reconstructed_y, y_i)
            
            # Use log scaling to avoid numerical instability
            rnn_loss_scaled = torch.log1p(rnn_loss * 10)
            # inv_loss_scaled = torch.log1p(inv_loss * 10)
            koopman_loss_scaled = torch.log1p(koopman_loss * 10)

            # Combine losses
            # loss = rnn_loss_scaled + 0.5 * inv_loss_scaled 
            loss = rnn_loss_scaled + koopman_loss_scaled
            
            return loss, rnn_loss, inv_loss, koopman_loss
            
        except Exception as e:
            print(f"Loss computation error: {str(e)}")
            raise
    
    def _masked_mse_loss(self, output, target, lengths):
        """Compute MSE loss considering sequence lengths"""
        try:
            # Create mask
            batch_size, seq_len, features = output.size()
            mask = torch.zeros_like(output)
            
            for i in range(batch_size):
                if lengths.numel() > i:  # Ensure enough elements in lengths
                    length = min(int(lengths[i].item()), seq_len)
                    if length > 0:  # Ensure positive length
                        mask[i, :length, :] = 1.0
            
            # Apply mask and compute loss
            # squared_diff = F.smooth_l1_loss(output * mask, target * mask, reduction='none', beta=5e-3)
            squared_diff = F.mse_loss(output * mask, target * mask, reduction='none')

            # Average
            valid_elements = mask.sum()
            if valid_elements > 0:
                loss = squared_diff.sum() / valid_elements
                return loss
            else:
                # If no valid elements, return zero loss
                return torch.tensor(0.0, device=self.device)

        except Exception as e:
            print(f"Masked MSE loss computation error: {str(e)}")
            raise
    
    def on_train_epoch_start(self):
        """Update current epoch and look_ahead at the start of each training epoch"""
        super().on_train_epoch_start() if hasattr(super(), 'on_train_epoch_start') else None
        
        # Progressive update of prediction steps
        if self.progressive_steps and hasattr(self.trainer, "max_epochs"):
            max_epochs = min(self.trainer.max_epochs, 60)
            # Gradually increase look_ahead during the first 60% of training
            progress_fraction = min(1.0, self.current_epoch / (max_epochs * 0.60))
            self.current_look_ahead = int(self.min_look_ahead + progress_fraction * (self.max_look_ahead - self.min_look_ahead))

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 将参数分组
        koopman_params = list(self._koopman_propagator.parameters())
        inn_params = list(self._inn.parameters())

        linear_layer_params = list(self._linear_layer.parameters())
        """
        # 使用不同的学习率
        optimizer = torch.optim.AdamW([
            {'params': inn_params, 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': koopman_params, 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': linear_layer_params, 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay}
        ])


        #Allinone
        optimizer = torch.optim.AdamW([
            {'params': inn_params, 'lr': 0.0007, 'weight_decay': 0.00004},
            {'params': koopman_params, 'lr': 0.0007 , 'weight_decay': 0.00004},
            {'params': linear_layer_params, 'lr': 0.0007, 'weight_decay': 0.00004}
        ])

        #iResNet
        optimizer = torch.optim.AdamW([
            {'params': inn_params, 'lr': 0.0006, 'weight_decay': 0.000089},
            {'params': koopman_params, 'lr': 0.0006 , 'weight_decay': 0.000089},
            {'params': linear_layer_params, 'lr': 0.0006, 'weight_decay': 0.000089}
        ])

        #Hint
        optimizer = torch.optim.AdamW([
            {'params': inn_params, 'lr': 0.00036, 'weight_decay': 0.000046},
            {'params': koopman_params, 'lr': 0.00036 , 'weight_decay': 0.000046},
            {'params': linear_layer_params, 'lr': 0.00036, 'weight_decay': 0.000046}
        ])

        
        #RealNVP
        optimizer = torch.optim.AdamW([
            {'params': inn_params, 'lr': 0.00059, 'weight_decay': 0.000092},
            {'params': koopman_params, 'lr': 0.00059 , 'weight_decay': 0.000092},
            {'params': linear_layer_params, 'lr': 0.00059, 'weight_decay': 0.000092}
        ])

        #Nice
        optimizer = torch.optim.AdamW([
            {'params': inn_params, 'lr': 0.00079, 'weight_decay': 0.000027},
            {'params': koopman_params, 'lr': 0.00079 , 'weight_decay': 0.000027},
            {'params': linear_layer_params, 'lr': 0.00079, 'weight_decay': 0.000027}
        ])

        #Glow
        optimizer = torch.optim.AdamW([
            {'params': inn_params, 'lr': 0.0009, 'weight_decay': 0.000098},
            {'params': koopman_params, 'lr': 0.0009 , 'weight_decay': 0.000098},
            {'params': linear_layer_params, 'lr': 0.0009, 'weight_decay': 0.000098}
        ])
        """
        
        """
        #ODE
        optimizer = torch.optim.AdamW([
            {'params': inn_params, 'lr': 0.00016, 'weight_decay': 0.00007},
            {'params': koopman_params, 'lr': 0.0016 , 'weight_decay': 0.00007},
            {'params': linear_layer_params, 'lr': 0.0016, 'weight_decay': 0.00007}
        ])
        """
        
        
        
        # 学习率调度器
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20
            ),
            "monitor": "loss",
            "interval": "epoch",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        """Define training step"""
        loss, rnn_loss, inv_loss, koopman_loss = self._compute_loss(batch)
        
        # Create loss dict for debugging
        loss_dict = {
            "rnn_loss": rnn_loss,
            "inv_loss": inv_loss,
            "koopman_loss": koopman_loss,
            "total_loss": loss
        }
        
        # Add gradient hooks for debugging
        if hasattr(self, 'debug_mode') and self.debug_mode:
            self._add_gradient_debug_hooks(loss_dict)

        # Log losses
        self.log("loss", loss, prog_bar=True)
        self.log("rnn_loss", rnn_loss, prog_bar=True)
        self.log("inv_loss", inv_loss, prog_bar=True)
        self.log("koop_loss", koopman_loss, prog_bar=True)
        self.log("epoch", float(self.current_epoch), prog_bar=True)
        self.log("look_ahead", float(self.current_look_ahead), prog_bar=True)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
        
        # Return loss dict
        return {
            "loss": loss,
            "rnn_loss": rnn_loss,
            "inv_loss": inv_loss,
            "koopman_loss": koopman_loss,
        }

    def _add_gradient_debug_hooks(self, loss_dict):
        """Add gradient debug hooks on key loss components"""
        if self.global_step % 500 == 0:  # Check every 500 steps
            for loss_name, loss_value in loss_dict.items():
                if not loss_value.requires_grad:
                    continue
                    
                def hook_generator(name):
                    def hook(grad):
                        print(f"Gradient norm for loss {name}: {grad.norm().item():.6f}")
                        if grad.isnan().any():
                            print(f"Warning: Gradient for loss {name} contains NaN!")
                        if grad.isinf().any():
                            print(f"Warning: Gradient for loss {name} contains Inf!")
                    return hook
                    
                loss_value.register_hook(hook_generator(loss_name))

    def on_after_backward(self):
        """Monitor gradients to ensure correct flow under different training modes"""
        # Check gradients every certain steps
        if self.global_step % 50 == 0:
            # Compute gradient norms for parameter groups
            koopman_grad_norm = self._get_param_group_grad_norm(self._koopman_propagator.parameters())
            inn_grad_norm = self._get_param_group_grad_norm(self._inn.parameters())
            extension_grad_norm = 0.0
            if self._inn.input_size < self._inn.output_size:
                extension_grad_norm = self._get_param_group_grad_norm(INNFactory.get_extension_params(self._inn))
            
            linear_layer_grad_norm = 0.0
            if self._linear_layer is not None:
                linear_layer_grad_norm = self._get_param_group_grad_norm(self._linear_layer.parameters())
            
            # Print gradient info
            print(f"Epoch {self.current_epoch}, Step {self.global_step}")
            print(f"  Koopman grad norm: {koopman_grad_norm:.6f}")
            print(f"  INN grad norm: {inn_grad_norm:.6f}")
            print(f"  Extension grad norm: {extension_grad_norm:.6f}")
            print(f"  Linear grad norm: {linear_layer_grad_norm:.6f}")

            # Log to TensorBoard
            self.log("grad_norm/koopman", koopman_grad_norm)
            self.log("grad_norm/inn", inn_grad_norm)
            self.log("grad_norm/extension", extension_grad_norm)
            self.log("grad_norm/linear", linear_layer_grad_norm)

    def _get_param_group_grad_norm(self, parameters):
        """Compute gradient norm for a parameter group"""
        total_norm = 0.0
        param_count = 0
        
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
        
        return (total_norm / max(1, param_count)) ** 0.5

    def check_parameter_updates(self):
        """Check if parameters are actually updated during training"""
        # Initialize parameter snapshots on first call
        if not hasattr(self, '_param_snapshots'):
            self._param_snapshots = {}
            for name, param in self.named_parameters():
                self._param_snapshots[name] = param.detach().clone()
            return
        
        # Calculate parameter changes
        koopman_changes = []
        inn_changes = []
        extension_changes = []
        linear_layer_changes = []
        
        for name, param in self.named_parameters():
            if name in self._param_snapshots:
                # Calculate parameter change
                delta = (param.detach() - self._param_snapshots[name]).abs().mean().item()
                
                # Collect changes by group
                if "_koopman_propagator" in name:
                    koopman_changes.append(delta)
                elif "_inn" in name:
                    inn_changes.append(delta)
                elif "_extension" in name:
                    extension_changes.append(delta)
                elif "_linear_layer" in name:
                    linear_layer_changes.append(delta)   
                    
                # Update snapshot
                self._param_snapshots[name] = param.detach().clone()
        
        # Compute average changes
        avg_koopman = sum(koopman_changes) / max(1, len(koopman_changes))
        avg_inn = sum(inn_changes) / max(1, len(inn_changes))
        avg_extension = sum(extension_changes) / max(1, len(extension_changes))
        avg_linear = sum(linear_layer_changes) / max(1, len(linear_layer_changes))
        
        # Print results
        print(f"Parameter update statistics (Epoch {self.current_epoch}):")
        print(f"  Koopman avg param change: {avg_koopman:.8f}")
        print(f"  INN avg param change: {avg_inn:.8f}")
        print(f"  Extension avg param change: {avg_extension:.8f}")
        print(f"  Linear avg param change: {avg_linear:.8f}")

    def visualize_gradient_flow(self, save_path="gradient_flow.png"):
        """Visualize gradient flow"""
        try:
            import matplotlib.pyplot as plt
            
            # Collect parameter and gradient data
            named_parameters = [(name, p) for name, p in self.named_parameters() if p.requires_grad and p.grad is not None]
            
            if not named_parameters:
                print("No gradients found to visualize")
                return
                
            plt.figure(figsize=(12, 8))
            
            # Group by parameter name
            param_groups = {'koopman': [], 'inn': [], 'extension': [], 'linear_layer': []}
            
            for name, param in named_parameters:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                relative_norm = grad_norm / (param_norm + 1e-8)
                
                if "_koopman_propagator" in name:
                    param_groups['koopman'].append((name, grad_norm, relative_norm))
                elif "_inn" in name:
                    param_groups['inn'].append((name, grad_norm, relative_norm))
                elif "_extension" in name:
                    param_groups['extension'].append((name, grad_norm, relative_norm))
                elif "_linear_layer" in name:
                    param_groups['linear_layer'].append((name, grad_norm, relative_norm)) 
            
            # Plot gradient distribution
            plt.subplot(2, 1, 1)
            for group_name, group_data in param_groups.items():
                if not group_data:
                    continue
                    
                x = range(len(group_data))
                y = [item[1] for item in group_data]  # Gradient norm
                plt.semilogy(x, y, 'o-', label=f'{group_name} (avg: {sum(y)/len(y):.4e})')
                
            plt.title('Parameter Gradient Norm Distribution')
            plt.xlabel('Parameter Index')
            plt.ylabel('Gradient Norm (log scale)')
            plt.legend()
            plt.grid(True)
            
            # Plot relative gradient strength
            plt.subplot(2, 1, 2)
            for group_name, group_data in param_groups.items():
                if not group_data:
                    continue
                    
                x = range(len(group_data))
                y = [item[2] for item in group_data]  # Relative gradient norm
                plt.semilogy(x, y, 'o-', label=f'{group_name} (avg: {sum(y)/len(y):.4e})')
                
            plt.title('Parameter Relative Gradient Strength (Gradient Norm / Param Norm)')
            plt.xlabel('Parameter Index')
            plt.ylabel('Relative Gradient Strength (log scale)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Gradient flow visualization saved to: {save_path}")
            
        except Exception as e:
            print(f"Error visualizing gradients: {str(e)}")


class DeepHODMD(BaseRegressor):
    """
    Enhanced Koopman Decomposition Model

    Parameters
    ----------
    dt : float, default=1.0
        Time step
    look_forward : int, default=10
        Steps for long-term prediction
    time_delay : int, default=1
        Steps for time-delay embedding
    config_inn : Dict[str, Any]
        Invertible neural network configuration
    batch_size : int, default=16
        Batch size
    normalize : bool, default=True
        Whether to normalize data
    normalize_mode : str, default="equal"
        Normalization mode
    normalize_std_factor : float, default=2.0
        Standard deviation scaling factor
    extension_hidden_size : int, default=128
        Extension layer hidden size
    extension_num_layers : int, default=5
        Number of layers in the extension layer
    progressive_steps : bool, default=True
        Whether to use progressive prediction steps
    trainer_kwargs : Dict[str, Any], default={}
        Trainer configuration
    """
    def __init__(
        self,
        dt: float = 1.0,
        look_forward: int = 10,
        time_delay: int = 1,
        config_inn: Dict[str, Any] = dict(input_size=10, hidden_size=32, num_layers=4, output_size=20),
        batch_size: int = 16,
        normalize: bool = True,
        normalize_mode: str = "equal",
        normalize_std_factor: float = 2.0,
        extension_config: Optional[Dict[str, Any]] = None,
        progressive_steps: bool = True,
        trainer_kwargs: Dict[str, Any] = {}
    ):
        # Required parameter check
        required_keys = ["input_size", "hidden_size", "num_layers", "output_size"]
        for key in required_keys:
            if key not in config_inn:
                raise ValueError(f"config_inn must contain '{key}' key")
        
        # Add default training parameters
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 0.5,
            "log_every_n_steps": 10
        }
        for key, value in default_trainer_kwargs.items():
            if key not in trainer_kwargs:
                trainer_kwargs[key] = value
        
        # Save configuration
        self.look_forward = look_forward
        self.time_delay = time_delay
        self.config_inn = config_inn
        self.normalize = normalize
        self.normalize_mode = normalize_mode
        self.dt = dt
        self.trainer_kwargs = trainer_kwargs
        self.normalize_std_factor = normalize_std_factor
        self.batch_size = batch_size
        self.extension_config = extension_config
        self.progressive_steps = progressive_steps

        

        
        # Initialize model
        self._regressor = DeepKoopmanRegressor(
            dt=dt,
            look_forward=look_forward,
            config_inn=config_inn,
            extension_config = extension_config,
            progressive_steps=progressive_steps
        )

    def _setup_trainer(self, monitor_params=False, log_dir='newkoopman_params', 
                      save_every_n_epochs=5, record_losses=True):
        """
        Setup Lightning trainer

        Parameters
        ----------
        monitor_params : bool
            Whether to monitor parameters
        log_dir : str
            Logging directory
        save_every_n_epochs : int
            Save frequency
        record_losses : bool
            Whether to record losses

        Returns
        -------
        L.Trainer
            Configured trainer
        """
        # Prepare callback list
        callbacks = []
        
        # Set parameter monitor
        if monitor_params:
            os.makedirs(log_dir, exist_ok=True)
            # Add parameter monitoring callback
            koopman_monitor = KoopmanParameterMonitor(
                log_dir=log_dir,
                save_every_n_epochs=save_every_n_epochs
            )
            callbacks.append(koopman_monitor)
        
        # Add loss recorder
        if record_losses:
            loss_recorder = LossRecorder(log_dir=log_dir)
            callbacks.append(loss_recorder)
        
        # Add early stopping callback
        early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
            monitor="loss",
            min_delta=5e-5,
            patience=30,
            verbose=True,
            mode="min"
        )
        callbacks.append(early_stop_callback)
        
        # Add model checkpoint callback
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            monitor="loss",
            dirpath=log_dir,
            filename="newkoopman-{epoch:02d}-{loss:.4f}",
            save_top_k=3,
            mode="min"
        )
        callbacks.append(checkpoint_callback)
        
        # Configure trainer
        trainer_config = {
            "callbacks": callbacks,
            "accelerator": "auto",
            "devices": 1,
        }
        
        # Merge user-provided config
        trainer_config.update(self.trainer_kwargs)
        
        # Create trainer
        trainer = L.Trainer(**trainer_config)
        
        return trainer
    
    def _train_model(self):
        """
        Execute model training

        Returns
        -------
        bool
            Whether training succeeded
        """
        try:
            # Start training
            self.trainer.fit(self._regressor, self.dm)
            
            # Ensure model switches to eval mode
            self._regressor.eval()
            
            # Check if training was interrupted
            if hasattr(self.trainer, 'interrupted') and self.trainer.interrupted:
                print("Training was interrupted by user")
                return False
            
            # Training succeeded
            return True
        
        except KeyboardInterrupt:
                print("Training was manually interrupted by user")
                # Ensure model switches to eval mode
                self._regressor.eval()
                # Try to compute model properties to retain current results
                try:
                    self._calculate_model_properties()
                except Exception as calc_err:
                    print(f"Error when computing interrupted model properties: {str(calc_err)}")
                return True  # Still considered successful, retain partially trained model

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return False
    
    def fit(self, x: Union[np.ndarray, List[np.ndarray]], 
            y: Optional[Union[np.ndarray, List[np.ndarray]]] = None, 
            dt: Optional[float] = None,
            monitor_params: bool = False,
            log_dir: str = 'newkoopman_params',
            save_every_n_epochs: int = 5,
            record_losses: bool = True,
            debug_mode: bool = False) -> "DeepHODMD":
        """
        Train the model

        Parameters
        ----------
        x : Union[np.ndarray, List[np.ndarray]]
            Input data or list of trajectories
        y : Optional[Union[np.ndarray, List[np.ndarray]]]
            Target data or list of trajectories
        dt : Optional[float]
            Override time step
        monitor_params : bool, default=False
            Whether to monitor Koopman parameters
        log_dir : str, default='newkoopman_params'
            Directory for parameter logging and visualization
        save_every_n_epochs : int, default=5
            Frequency to save parameter states
        record_losses : bool, default=True
            Whether to record detailed losses

        Returns
        -------
        NovelHODMD
            Trained model
        """

        # Set debug mode
        if debug_mode:
            # Enable anomaly detection for better gradient diagnostics
            torch.autograd.set_detect_anomaly(True)
            self._regressor.debug_mode = True
            print("Gradient debug mode enabled")

        # Progress feedback
        print("Starting NovelHODMD model training...")
        
        try:
            # Prepare data
            self.n_input_features_ = self.config_inn["input_size"]
            self.dm = self._prepare_data(x, y, dt)
            
            # Setup trainer
            self.trainer = self._setup_trainer(
                monitor_params,
                log_dir,
                save_every_n_epochs,
                record_losses
            )
            
            # Train model
            training_success = self._train_model()
            
            if training_success:
                # Compute model properties
                self._calculate_model_properties()
            
            # Disable anomaly detection for better performance
            torch.autograd.set_detect_anomaly(False)
            
            print("Model training completed!")
            return self
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            torch.autograd.set_detect_anomaly(False)
            raise

    def _calculate_model_properties(self):
        """Compute and store model properties after training"""
        try:
            # Compute Koopman operator information
            _state_matrix_, _eigenvalues_ ,_eigenvectors_, _ = self._regressor._koopman_propagator.get_discrete_time_Koopman_Operator()
            self._state_matrix_, self._eigenvalues_, self._eigenvectors_ = (
                _state_matrix_.detach().numpy(), 
                _eigenvalues_.detach().numpy(), 
                _eigenvectors_.detach().numpy()
            )
            # self._eigenvalues_, self._eigenvectors_ = np.linalg.eig(self._state_matrix_)
            self._coef_ = self._state_matrix_.copy()
            
            # Safely compute Jacobian matrix
            try:
                # Sample data points
                sample_points = []
                with torch.no_grad():
                    for batch in self.dm.train_dataloader():
                        x_batch = batch[0]
                        if self.normalize:
                            x_batch = self.dm.normalization(x_batch)
                        phi_batch = self._regressor._encode(x_batch)
                        sample_points.append(phi_batch)
                        if len(sample_points) >= 5:
                            break
                    
                    # Get average encoded point
                    encoded_points = torch.cat(sample_points, dim=0)
                    mean_point_data = encoded_points.mean(dim=0, keepdim=True).clone().detach()
                
                # Create tensor with gradient requirement
                mean_point = mean_point_data.clone().detach().requires_grad_(True)
                
                # Switch to eval mode but keep gradients
                self._regressor.eval()
                
                # Compute Jacobian - this computation may be unnecessary
                jacobian_rows = []
                for i in range(self.config_inn["input_size"]):
                    try:
                        # Compute decoded value
                        x_decoded = self._regressor._decode(mean_point)
                        x_decoded_i = x_decoded[0, i]
                        
                        # Compute gradient
                        grad_i = torch.autograd.grad(
                            x_decoded_i, 
                            mean_point, 
                            retain_graph=True, 
                            create_graph=False,
                            allow_unused=True
                        )[0]
                        
                        # Ensure gradient is not None
                        if grad_i is None:
                            # If gradient is None, fill with zeros
                            jacobian_rows.append(torch.zeros_like(mean_point).reshape(-1))
                        else:
                            # Add gradient vector to Jacobian
                            jacobian_rows.append(grad_i.reshape(-1))
                    
                    except Exception as e:
                        print(f"Error computing Jacobian row {i}: {str(e)}")
                        # Add zero vector as fallback
                        jacobian_rows.append(torch.zeros_like(mean_point).reshape(-1))
                
                # # Assemble full Jacobian matrix
                jacobian_matrix = torch.stack(jacobian_rows, dim=0)
                self._jacobian_matrix = jacobian_matrix.detach().cpu().numpy()

                # Stack rows to form Jacobian
                if jacobian_rows:
                    self._ur = np.vstack(jacobian_rows)
                    if self.normalize:
                        std = self.dm.inverse_transform.std
                        self._ur = np.diag(std) @ self._ur

                    self._unnormalized_modes = self._ur @ self._eigenvectors_
                else:
                    # If Jacobian cannot be computed, create identity as fallback
                    print("Warning: Could not compute Jacobian matrix")
                    self._ur = np.eye(self.config_inn["input_size"])
                    self._unnormalized_modes = self._eigenvectors_.copy()


                # Restore training mode
                self._regressor.train()
                
            except Exception as e:
                print(f"Error computing Jacobian: {str(e)}")
                self._jacobian_matrix = None
                
        except Exception as e:
            print(f"Error computing model properties: {str(e)}")
            # Set default values
            self._state_matrix_ = None
            self._eigenvalues_ = None
            self._eigenvectors_ = None
            self._coef_ = None
            self._jacobian_matrix = None
      
    def _prepare_data(self, x, y=None, dt=None, original_look_forward=None):
        """
        Prepare data for training by creating appropriate data module

        Parameters
        ----------
        x : Union[np.ndarray, List[np.ndarray]]
            Input data or list of trajectories
        y : Union[np.ndarray, List[np.ndarray]], optional
            Target data or list of trajectories
        dt : float, optional
            Time step override
        original_look_forward : int, optional
            Override the default look_forward value

        Returns
        -------
        DataModule
            Initialized and prepared data module
        """
        look_forward = original_look_forward if original_look_forward is not None else self.look_forward
        
        # Override dt if provided
        if dt is not None:
            self.dt = dt
            if hasattr(self, '_regressor'):
                self._regressor._koopman_propagator.dt = torch.tensor(dt)
        
        # Create the data module based on input types
        # Case 1: a single traj, x is 2D np.ndarray, no validation
        if y is None and isinstance(x, np.ndarray) and x.ndim == 2:
            t0, t1 = x[:-1], x[1:]
            list_of_traj = [np.stack((t0[i], t1[i]), 0) for i in range(len(x) - 1)]
            dm = TimeDalaySeqDataModule(
                list_of_traj,
                None,
                look_forward,
                self.time_delay,
                self.batch_size,
                self.normalize,
                self.normalize_mode,
                self.normalize_std_factor,
            )
            self.n_samples_ = len(list_of_traj)
        
        # Case 2: x, y are 2D np.ndarray, no validation
        elif (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and 
                x.ndim == 2 and y.ndim == 2):
            t0, t1 = x, y
            list_of_traj = [np.stack((t0[i], t1[i]), 0) for i in range(len(x) - 1)]
            dm = TimeDalaySeqDataModule(
                list_of_traj,
                None,
                look_forward,
                self.time_delay,
                self.batch_size,
                self.normalize,
                self.normalize_mode,
                self.normalize_std_factor,
            )
            self.n_samples_ = len(list_of_traj)

        # Case 3: only training data, x is a list of trajectories, y is None
        elif isinstance(x, list) and y is None:
            dm = TimeDalaySeqDataModule(
                x,
                None,
                look_forward,
                self.time_delay,
                self.batch_size,
                self.normalize,
                self.normalize_mode,
                self.normalize_std_factor,
            )
            self.n_samples_ = len(x)
        
        # Case 4: both training and validation data are provided as lists
        elif isinstance(x, list) and isinstance(y, list):
            dm = TimeDalaySeqDataModule(
                x,
                y,
                look_forward,
                self.time_delay,
                self.batch_size,
                self.normalize,
                self.normalize_mode,
                self.normalize_std_factor,
            )
            self.n_samples_ = len(x)
        else:
            raise ValueError("Incorrect format for x and y, please check input data")
        
        # Prepare the data
        dm.prepare_data()
        
        # Setup the data module
        try:
            # Explicitly pass 'fit' as the stage parameter
            dm.setup(stage='fit')
        except Exception as e:
            try:
                # Try with positional argument
                dm.setup('fit')
            except Exception as e:
                # Fall back to no parameter
                dm.setup()
                
        return dm

    def predict(self, x, n_steps=1, return_sequences=True):
        """
        Predict future states

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input initial state, shape (n_samples, n_features) or (n_features,)
        n_steps : int, default=1
            Number of prediction steps
        return_sequences : bool, default=True
            Whether to return the whole sequence or only the last step

        Returns
        -------
        np.ndarray
            Prediction results
        """
        # Ensure x has proper shape
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = np.array([x])
            x_tensor = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            if x.dim() == 1:
                x_tensor = x.reshape(1, -1)
            else:
                x_tensor = x
        else:
            raise ValueError("Input must be a numpy array or pytorch tensor")
        
        # Apply normalization
        if self.normalize and hasattr(self.dm, 'normalization'):
            x_tensor = self.dm.normalization(x_tensor)

        
        # Set device
        device = next(self._regressor.parameters()).device
        x_tensor = x_tensor.to(device)

        
        # Perform prediction
        with torch.no_grad():
            pred = self._regressor.forward(x_tensor, n=n_steps, record=return_sequences)

        
        # Convert back to numpy
        if return_sequences:
            pred_np = pred.cpu().numpy()
        else:
            pred_np = pred[:, -1, :].cpu().numpy()
        
        # Squeeze first dimension if only one sample
        if pred_np.shape[0] == 1 and x.shape[0] == 1:
            pred_np = pred_np.squeeze(0)
        
        return pred_np

    @torch.no_grad()
    def simulate(self, x: Union[np.ndarray, torch.Tensor], 
                 n_steps: int) -> np.ndarray:
        """
        Simulate the system forward in time for `n_steps` steps starting from `x`.

        Args:
            x (np.ndarray or torch.Tensor): The initial state of the system.
            Should be a 2D array/tensor.
            n_steps (int): The number of time steps to simulate the system forward.

        Returns:
            np.ndarray: The simulated states of the system. Will be of shape
            `(n_steps+1, n_features)`.
        """
        self._regressor.eval()
        x = self._convert_input_ndarray_to_tensor(x)
        x0 = x.clone()
        dim_preserve = x.ndim
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            if self.normalize:
                x = self.dm.normalization(x)
                x_future = self._regressor(x, n_steps, record=True)
                x_future = self.dm.inverse_transform(x_future)
            else:
                x_future = self._regressor(x, n_steps, record=True)
            
            if dim_preserve == 2:
                x_future = x_future.squeeze(0)
            x_future = torch.cat([x0, x_future], dim=-2)
            return x_future.numpy()

    def koopman_reconstruct(self, x, t=1):
        """
        Reconstruct state at time t using Koopman decomposition

        Args:
            x (np.ndarray): Initial state, Input data 
                of shape `(n_samples, n_features) or (n_features, )`.
            t (int): Time steps to advance

        Returns:
            np.ndarray: Reconstructed state at time t
                of shape '(n_samples, n_features)'
        """
        # Compute initial psi (eigenfunction coefficients)
        psi_0 = self._compute_psi(x)                                # (n_koopman, n_samples)
        
        # Advance using eigenvalues
        psi_t = np.diag(np.power(self._eigenvalues_, t)) @ psi_0    # (n_koopman, n_samples)
        
        # Convert back to observables
        phi_t = self._eigenvectors_ @ psi_t                         # (n_koopman, n_samples)

        # Decode to state space using nonlinear decoder
        x_t = self.reconstruct_x(phi_t)                            # (n_samples, n_features)    
        
        return x_t

    def get_koopman_modes(self):
        """
        Get Koopman eigenfunctions and eigenvalues

        Returns
        -------
        dict
            Dictionary containing Koopman eigenfunctions and eigenvalues
        """
        if not hasattr(self, '_eigenvalues_') or self._eigenvalues_ is None:
            raise ValueError("Model not trained or property computation failed")
        
        # Prepare result dict
        modes_info = {
            'eigenvalues': self._eigenvalues_,
            'eigenvectors': self._eigenvectors_,
            'koopman_operator': self._state_matrix_
        }
        
        # Add Jacobian if available
        if hasattr(self, '_jacobian_matrix') and self._jacobian_matrix is not None:
            modes_info['jacobian'] = self._jacobian_matrix
        
        return modes_info

    def _compute_phi(self, x):
        """
        Computes the Koopman observable vector `phi(x)` for input `x`.

        Args:
            x (np.ndarray): The input state vector. 
                shape: (n_samples, n_features) or (n_features, )

        Returns:
            np.ndarray: The Koopman observable vector `phi(x)` for input `x`. 
                shape: `(n_koopmanm, n_samples)`
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        self._regressor.eval()
        x = self._convert_input_ndarray_to_tensor(x)

        if self.normalize:
            x = self.dm.normalization(x)
        phi = self._regressor._encode(x).detach().numpy()
        return phi.T

    def _compute_psi(self, x):
        """
        Computes the Koopman eigenfunction expansion coefficients `psi(x)` given `x`.

        Args:
            x (numpy.ndarray): Input data
                of shape '(n_samples, n_features) or (n_features, )'.

        Returns:
            numpy.ndarray: Koopman eigenfunction expansion coefficients `psi(x)`
                of shape `(n_koopman, n_samples)`.
        """
        if x.ndim == 1:
            x = x.reshape(1,-1)

        # Get observables
        phi = self._compute_phi(x)
        
        # Project onto eigenvectors
        psi = np.linalg.inv(self._eigenvectors_) @ phi
        return psi
    
    def reverse_phi(self, phi):
        """
        Reconstruct the original state using the full nonlinear decoder
        Args:
            phi_col (np.ndarray): Encoded vector, shape (n_koopman, n_samples) or (n_samples,)
        Returns:
            np.ndarray: Reconstructed original state, shape (n_samples, n_features)
        """
        if phi.ndim == 1:
            phi = phi.reshape(-1, 1) # (n_koopman, 1)
        phi = torch.FloatTensor(phi) # Transpose to match model input format
        self._regressor.eval()
        with torch.no_grad():
            # Use the inverse function of the invertible network for nonlinear decoding
            phi_recon = self._regressor._decode(phi.T) # (n_samples, n_features)
            if self.normalize:
                phi_recon = self.dm.inverse_transform(phi_recon)
        return phi_recon.numpy()

    def visualize_modes(self, save_path=None):
        """
        Visualize Koopman modes

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot

        Returns
        -------
        None
        """
        try:
            import matplotlib.pyplot as plt
            
            if not hasattr(self, '_eigenvalues_') or self._eigenvalues_ is None:
                raise ValueError("Model not trained or property computation failed")
            
            # Plot eigenvalue distribution
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            
            # Eigenvalue complex plane
            ax = axes[0, 0]
            ev_real = np.real(self._eigenvalues_)
            ev_imag = np.imag(self._eigenvalues_)
            sc = ax.scatter(ev_real, ev_imag, s=100, alpha=0.7, c=np.abs(self._eigenvalues_))
            ax.set_title('Eigenvalue Distribution in Complex Plane')
            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            # Unit circle
            circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)
            ax.grid(True)
            plt.colorbar(sc, ax=ax, label='Magnitude')
            
            # Eigenvalue magnitude distribution
            ax = axes[0, 1]
            magnitudes = np.abs(self._eigenvalues_)
            index = np.arange(len(magnitudes))
            sorted_idx = np.argsort(magnitudes)[::-1]
            ax.bar(index, magnitudes[sorted_idx])
            ax.set_title('Eigenvalue Magnitude Distribution (Descending)')
            ax.set_xlabel('Index')
            ax.set_ylabel('Magnitude')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            ax.grid(True)
            

            # Show Koopman matrix heatmap
            ax = axes[1, 0]
            im = ax.imshow(np.abs(self._state_matrix_), cmap='viridis')
            ax.set_title('Koopman Operator Absolute Value Heatmap')
            plt.colorbar(im, ax=ax)
            
            # Top important eigenvectors
            ax = axes[1, 1]
            top_idx = np.argsort(magnitudes)[-5:][::-1]
            for i, idx in enumerate(top_idx):
                ev = self._eigenvectors_[:, idx]
                ax.plot(np.abs(ev), label=f'λ={self._eigenvalues_[idx]:.2f}')
            ax.set_title('Top 5 Important Eigenvectors (by Eigenvalue Magnitude)')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is not None:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing modes: {str(e)}")
            import traceback
            traceback.print_exc()
    
    @property
    def A(self):
        """Returns the state transition matrix A of the INNDMD model.

        Returns
        -------
        A : numpy.ndarray
            The state transition matrix of shape (n_states, n_states), where
            n_states is the number of states in the model.
        """
        return self._state_matrix_

    def W(self, psi_t):
        """
        Returns the x_t using the Koopman modes self._eigenvectors_ from Koopman eigenvalues psi_t.
        
        Input:
        psi_t: (n_koopman, n_samples)

        Returns:
        --------
        numpy.ndarray of shape (n_time, n_x)
            The matrix V, where each column represents a Koopman mode.
        """
        # Convert back to observables
        phi_t = self._eigenvectors_ @ psi_t   # shape (n_koopman, n_samples) or (n_samples,)

        # Decode to state space using nonlinear decoder
        x_t = self.reconstruct_x(phi_t)        # shape (n_samples, features)
    
        return x_t

    def phi(self, x):
        """
        Computes the Koopman observable vector `phi(x)` for input `x`.
        
        Args:
            x_col (np.ndarray): The input state vector.
                shape: (n_samples, n_features) or (n_features, )

        Returns:
            np.ndarray: The Koopman observable vector. 
                shape: (n_koopman, n_samples)
        """
        return self._compute_phi(x)

    def psi(self, x):
        """
        Computes the Koopman eigenfunction expansion coefficients `psi(x)` given `x`.
        
        Args:
            x (np.ndarray): The input state vector. 
                shape: (n_samples, n_features) or (n_features, )
            
        Returns:
            np.ndarray: The Koopman eigenfunction expansion coefficients. 
                shape: (n_koopman, n_samples)
        """
        return self._compute_psi(x)

    def _convert_input_ndarray_to_tensor(self, x):
        """
        Converts input numpy ndarray to PyTorch tensor with appropriate dtype and
        device.

        Args:
            x (np.ndarray or torch.Tensor): Input data as numpy ndarray or PyTorch
                tensor.

        Returns:
            torch.Tensor: Input data as PyTorch tensor.

        Raises:
            TypeError: If input data is not a numpy ndarray or PyTorch tensor.
            ValueError: If input array has more than 2 dimensions.
        """
        if isinstance(x, np.ndarray):
            if x.ndim > 2:
                raise ValueError("input array should be 1 or 2D")
            if x.ndim == 1:
                x = x.reshape(1, -1)
            # Create tensor but do not specify device
            x = torch.FloatTensor(x)
        elif isinstance(x, torch.Tensor):
            if x.ndim != 2:
                raise ValueError("input tensor `x` must be a 2d tensor")
        
        # In Lightning, device is handled automatically
        return x

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        return self._state_matrix_

    @property
    def eigenvalues_(self):
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def eigenvectors_(self):
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def unnormalized_modes(self):
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    @property
    def ur(self):
        check_is_fitted(self, "_ur")
        return self._ur


import pickle
from pykoopman.regression._nndmd import (TensorNormalize, InverseTensorNormalize)
from warnings import warn

class TimeDalaySeqDataModule(SeqDataModule):
    """
    Class for creating sequence data dataloader for training and validation.

    Args:
        data_tr: List of 2D numpy.ndarray representing training data trajectories.
        data_val: List of 2D numpy.ndarray representing validation data trajectories.
            Can be None.
        look_forward: Number of time steps to predict forward.
        batch_size: Size of each batch of data.
        normalize: Whether to normalize the input data or not. Default is True.
        normalize_mode: The type of normalization to use. Either "equal" or "max".
            Default is "equal".
        normalize_std_factor: Scaling factor for standard deviation during
            normalization. Default is 2.0.

    Methods:
        prepare_data(): Prepares the data by converting to time-delayed data and
            computing mean and std if normalize is True.
        setup(stage=None): Sets up training and validation datasets.
        train_dataloader(): Returns a DataLoader for training data.
        val_dataloader(): Returns a DataLoader for validation data.
        convert_seq_list_to_delayed_data(data_list, look_back, look_forward, time_delay): Converts
            list of sequences to time-delayed data.
        collate_fn(batch): Custom collate function to be used with DataLoader.

    Returns:
        A SeqDataModule object.
    """

    def __init__(
        self,
        data_tr,
        data_val,
        look_forward=10,
        time_delay = 1,
        batch_size=32,
        normalize=True,
        normalize_mode="equal",
        normalize_std_factor=2.0,
    ):
        """
        Initialize a SeqDataModule.

        Args:
            data_tr (Union[str, List[np.ndarray]]): Training data. Can be either a
                list of 2D numpy arrays, each 2D numpy array representing a trajectory,
                or the path to a pickle file containing such a list.
            data_val (Optional[Union[str, List[np.ndarray]]]): Validation data.
                Can be either a list of 2D numpy arrays, each 2D numpy array
                    representing a trajectory, or the path to a pickle file
                    containing such a list.
            look_forward (int): Number of time steps to predict into the future.
            batch_size (int): Number of samples per batch.
            normalize (bool): Whether to normalize the data. Default is True.
            normalize_mode (str): Mode for normalization. Can be either "equal"
                or "max". "equal" divides by the standard deviation, while "max"
                divides by the maximum absolute value of the data. Default is "equal".
            normalize_std_factor (float): Scaling factor for the standard deviation in
                normalization. Default is 2.0.

        Returns:
            None.
        """
        super().__init__(
            data_tr = data_tr,
            data_val = data_val,
            look_forward = look_forward,
            batch_size = batch_size,
            normalize = normalize,
            normalize_mode = normalize_mode,
            normalize_std_factor = normalize_std_factor,
            # input data_tr or data_val is a list of 2D np.ndarray. each 2d
            # np.ndarray is a trajectory, and the axis 0 is number of samples, axis 1 is
            # the number of system state
        )
        self.time_delay = time_delay


    def prepare_data(self):
        """
        Preprocesses the input training and validation data by checking their types,
        checking for normalization, finding the mean and standard deviation of
        the training data (if normalization is enabled), and creating time-delayed data
        from the input data.

        Raises:
            ValueError: If the training data is None or has an invalid type.
            ValueError: If the validation data has an invalid type.
            TypeError: If the data is complex or not float.

        """
        # train data
        if self.data_tr is None:
            raise ValueError("You must feed training data!")
        if isinstance(self.data_tr, list):
            data_list = self.data_tr
        elif isinstance(self.data_tr, str):
            f = open(self.data_tr, "rb")
            data_list = pickle.load(f)
        else:
            raise ValueError("Wrong type of `self.data_tr`")

        # check train data
        data_list = self.check_list_of_nparray(data_list)

        # time-delay embedding
        embedded_data_list = self.time_delay_embedding(data_list, self.time_delay)

        # find the mean, std
        if self.normalize:
            stacked_data_list = np.vstack(embedded_data_list)
            mean = stacked_data_list.mean(axis=0)
            std = stacked_data_list.std(axis=0)

            # zero mean so easier for downstream
            self.mean = torch.FloatTensor(mean) * 0
            # default = 2.0, more stable
            self.std = torch.FloatTensor(std) * self.normalize_std_factor

            if self.normalize_mode == "max":
                self.std = torch.ones_like(self.std) * self.std.max()

            # prevent divide by zero error
            for i in range(len(self.std)):
                if self.std[i] < 1e-6:
                    self.std[i] += 1e-3

            # get transform
            self.normalization = TensorNormalize(self.mean, self.std)

            # get inverse transform
            self.inverse_transform = InverseTensorNormalize(self.mean, self.std)

        # create time-delayed data
        self._tr_x, self._tr_yseq, self._tr_ys = self.convert_seq_list_to_delayed_data(
            embedded_data_list, self.look_back, self.look_forward
        )
        
        # validation data
        if self.data_val is not None:
            # raise ValueError("You need to feed validation data!")
            if isinstance(self.data_val, list):
                data_list = self.data_val
            elif isinstance(self.data_val, str):
                f = open(self.data_val, "rb")
                data_list = pickle.load(f)
            else:
                raise ValueError("Wrong type of `self.data_val`")

            # check val data
            data_list = self.check_list_of_nparray(data_list)
            # time-delay embedding
            embedded_data_list = self.time_delay_embedding(data_list, self.time_delay)
            # create time-delayed data
            self._val_x, self._val_yseq, self._val_ys = self.convert_seq_list_to_delayed_data(
                embedded_data_list, self.look_back, self.look_forward)
        else:
            warn("Warning: no validation data prepared")

    def time_delay_embedding(self, data_list, time_delay):
        """
        Converts a list of sequences to time-delayed data by extracting subsequences
        of length `time_delay` from each sequence in the list.

        Args:
            data_list (List[np.ndarray]): A list of 2D numpy arrays. Each array
                represents a trajectory, with axis 0 representing the number of samples
                and axis 1 representing the number of system states.
            time_delay (int): The number of steps for time-delay embedding.

        Returns:
            List[np.ndarray]: A list of embedded sequences. Each embedded sequence is a 2D numpy array
                of shape (time_length, features * time_delay).
        """
        # First, process time-delay embedding
        embedded_seq_list = []

        if time_delay <= 1:
            # If no delay embedding is needed, use the original sequences directly
            embedded_seq_list = data_list
        else:
            for seq in data_list:
                # Ensure the sequence is long enough
                if len(seq) < time_delay:
                    # If the sequence is too short for delay embedding, skip this sample
                    continue

                # Create time-delay embedded sequences
                embedded_seq = []
                for i in range(len(seq) - time_delay + 1):
                    # Collect consecutive time points and flatten into one feature vector
                    window = seq[i:i+time_delay].reshape(-1)
                    embedded_seq.append(window)
                    
                # Convert the embedded sequence to a numpy array
                embedded_seq = np.array(embedded_seq)
                embedded_seq_list.append(embedded_seq)

        return embedded_seq_list
