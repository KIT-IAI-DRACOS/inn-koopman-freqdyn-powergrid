from ._base_inn import BaseINN
import torch

class IdentityINN(BaseINN):
    """
    Identity Mapping
    Serves as Benchmark for Extensions
    """
    def __init__(self, input_size):
        super(IdentityINN, self).__init__(input_size)
        self.input_size = input_size

    def _check_input_features(self, x):
        if x.dim() == 3:
            # Überprüfe die Feature-Dimension, wenn 3D-Tensor (Batch, Seq_len, Features)
            if x.shape[2] != self.input_size:
                raise ValueError(f"Input feature dimension {x.shape[2]} doesn't match expected {self.input_size}")
        elif x.dim() == 2:
            # Überprüfe die Feature-Dimension, wenn 2D-Tensor (Batch*Seq_len, Features)
            if x.shape[1] != self.input_size:
                 raise ValueError(f"Input feature dimension {x.shape[1]} doesn't match expected {self.input_size}")
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, only 2D or 3D tensors are supported")


    def forward(self, x):
        # Feature-Dimension überprüfen, um die Schnittstellenkonformität sicherzustellen
        self._check_input_features(x)
        return x

    def inverse(self, z):
        # Feature-Dimension überprüfen
        self._check_input_features(z)
        return z