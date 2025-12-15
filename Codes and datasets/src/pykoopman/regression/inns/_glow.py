from ._base_inn import BaseINN
import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class FreiaGlowINN(BaseINN):
    """
    GLOW-style Invertible Neural Network based on FrEIA.
    Support for 2D and 3D input
    
    Parameter
    ----------
    input_size : int
        Feature dimension of input
    hidden_size : int
        Hidden Layer size in subnetwork of coupling block
    num_layers : int
        Number of invertible coupling blocks
    init_identity : bool
        initialize subnetwork
    """
    def __init__(self, input_size, hidden_size, num_layers, init_identity=True):
        super(FreiaGlowINN, self).__init__(input_size)
        self.input_size = input_size
        self.init_identity = init_identity

        nodes = [Ff.InputNode(input_size, name='input')]

        for i in range(num_layers):
            # Optional: ActNorm vor dem Coupling-Block
            #nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}, name=f'actnorm_{i}'))
            # Optional: Permutation, hier random als Beispiel
            #nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': i}, name=f'permute_{i}'))

            # Coupling Block mit einem Subnetz ähnlich deinem _subnet_fc
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {'subnet_constructor': self._subnet_fc(hidden_size, init_identity),
                     'clamp': 2.0},
                    name=f'coupling_{i}'
                )
            )

        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        self.net = Ff.GraphINN(nodes)

    def _subnet_fc(self, hidden_size, init_identity=True):
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
            if init_identity:
                torch.nn.init.zeros_(net[-1].weight)
                if net[-1].bias is not None:
                    torch.nn.init.zeros_(net[-1].bias)
            return net
        return subnet_fc

    def _process_2d(self, x, rev=False):
        if not rev:
            out, _ = self.net(x)
            return out
        else:
            x_rec, _ = self.net(x, rev=True)
            return x_rec

    def forward(self, x):
        if x.dim() == 2:
            return self._process_2d(x)
        elif x.dim() == 3:
            batch_size, seq_len, features = x.shape
            if features != self.input_size:
                raise ValueError(f"Input feature dimension {features} doesn't match expected {self.input_size}")
            x_reshaped = x.reshape(-1, features)
            out_2d = self._process_2d(x_reshaped)
            out_features = out_2d.shape[1]
            out_3d = out_2d.reshape(batch_size, seq_len, out_features)
            return out_3d
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, only 2D or 3D tensors are supported")

    def inverse(self, z):
        if z.dim() == 2:
            return self._process_2d(z, rev=True)
        elif z.dim() == 3:
            batch_size, seq_len, features = z.shape
            z_reshaped = z.reshape(-1, features)
            out_2d = self._process_2d(z_reshaped, rev=True)
            out_features = out_2d.shape[1]
            out_3d = out_2d.reshape(batch_size, seq_len, out_features)
            return out_3d
        else:
            raise ValueError(f"Unsupported input dimension: {z.dim()}, only 2D or 3D tensors are supported")
