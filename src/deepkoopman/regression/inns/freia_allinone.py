from ._base_inn import BaseINN
import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class FreiaINN(BaseINN):
    """
    Invertible Neural Network based on FrEIA library.
    Automatically handles 2D and 3D input tensors.
    
    Parameters
    ----------
    input_size : int
        Input feature dimension
    hidden_size : int
        Hidden layer size in the subnetworks
    num_layers : int
        Number of invertible coupling blocks
    output_size : int, optional
        Output feature dimension. If None, equals input_size
    """
    def __init__(self, input_size, hidden_size, num_layers, init_identity = True):
        
        super(FreiaINN, self).__init__(input_size)  # Der Parent-Konstruktor
        self.init_identity = init_identity
        self.input_size = input_size
        
        # Build invertible network
        nodes = [Ff.InputNode(input_size, name='input')]
        
        # Add invertible layers
        for i in range(num_layers):
            nodes.append(Ff.Node(nodes[-1],
                                Fm.AllInOneBlock,
                                {'subnet_constructor': self._subnet_fc(hidden_size, init_identity),
                                 'affine_clamping': 2.0},  # Limit the magnitude of affine transforms
                                name=f'layer_{i}'))
        
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        self.net = Ff.GraphINN(nodes)

    def _subnet_fc(self, hidden_size, init_identity=True):
        """Subnetwork constructor for coupling blocks"""
        def subnet_fc(dims_in, dims_out):
            net = torch.nn.Sequential(
                torch.nn.Linear(dims_in, hidden_size),
                torch.nn.LayerNorm(hidden_size),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LayerNorm(hidden_size),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Linear(hidden_size, dims_out)
            )
            
            # If initializing as identity, set the last layer's weights and bias to zero
            if init_identity:
                # In AllInOneBlock, zero output from subnet yields identity mapping
                torch.nn.init.zeros_(net[-1].weight)
                if net[-1].bias is not None:
                    torch.nn.init.zeros_(net[-1].bias)
            
            return net
        
        return subnet_fc
    
    def _process_2d(self, x, rev=False, ignore_projection=False):
        """Process 2D tensor input"""
        if not rev:
            # Forward propagation
            out, _ = self.net(x)
            return out
        else:
            # Inverse propagation
            x_rec, _ = self.net(x, rev=True)
            return x_rec
    
    def forward(self, x):
        """
        Forward propagation (encoding).
        Supports 2D tensors (batch_size or sequence_length, input_size) and 3D tensors (batch_size, sequence_length, input_size).
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
            out_2d = self._process_2d(x_reshaped)
            
            # Reshape back to 3D tensor (batch_size, seq_len, output_size)
            out_features = out_2d.shape[1]
            out_3d = out_2d.reshape(batch_size, seq_len, out_features)
            
            return out_3d
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, only 2D or 3D tensors are supported")
    
    def inverse(self, z):
        """
        Inverse propagation (decoding).
        Supports 2D and 3D tensors.
        """
        # Check input dimension
        if z.dim() == 2:
            # 2D input
            return self._process_2d(z, rev=True)
        elif z.dim() == 3:
            # 3D input
            batch_size, seq_len, features = z.shape
            
            # Reshape to 2D tensor
            z_reshaped = z.reshape(-1, features)
            
            # Process 2D tensor
            out_2d = self._process_2d(z_reshaped, rev=True)
            
            # Reshape back to 3D tensor
            out_features = out_2d.shape[1]
            out_3d = out_2d.reshape(batch_size, seq_len, out_features)
            
            return out_3d
        else:
            raise ValueError(f"Unsupported input dimension: {z.dim()}, only 2D or 3D tensors are supported")