import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class HybridINNWrapper(LightningModule):
    def __init__(self, base_inn, extension_net):
        super().__init__()
        self.base_inn = base_inn
        self.extension_net = extension_net

    @property
    def input_size(self):
        return self.base_inn.input_size

    @property
    def output_size(self):
        return self.base_inn.output_size + self.extension_net.output_size

    def forward(self, x):
        z_base = self.base_inn.forward(x)
        z_ext = self.extension_net(x)
        return torch.cat([z_base, z_ext], dim=-1)

    def inverse(self, phi):
        z_base = phi[:, :self.base_inn.output_size]
        return self.base_inn.inverse(z_base)
    
    def get_extension_params(self):
        if self.extension_net is not None:
            self.extension_net.get