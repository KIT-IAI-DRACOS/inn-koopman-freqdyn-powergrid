
import torch.nn as nn

class BaseINN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size 

    def forward(self, x):
        raise NotImplementedError("subclasses need to implement 'forward' method")

    def inverse(self, z):
        raise NotImplementedError("subclasses need to implement 'inverse' method")


