from typing import Dict
import torch.nn as nn

from deepkoopman.regression._extensionlayer import CNNExtensionLayer
from deepkoopman.regression._extensionlayer import KernelExtensionLayer
from deepkoopman.regression._extensionlayer import Extensionlayer  

class ExtensionFactory:
    @staticmethod
    def create_extension(name: str, input_size: int, output_size: int, config: Dict) -> nn.Module:
        name = name.lower()

        if name == "cnn":
            return CNNExtensionLayer(
                input_size=input_size,
                output_size=output_size,
                hidden_channels=config.get("hidden_channels", [32, 64, 128]),
                kernel_sizes=config.get("kernel_sizes", [3, 5, 7]),
                dropout_rate=config.get("dropout_rate", 0.1),
                use_residual=config.get("use_residual", True),
            )

        elif name == "kernel":
            return KernelExtensionLayer(
                input_size=input_size,
                output_size=output_size,
                kernel_type=config.get("kernel_type", "rbf"),
                kernel_params=config.get("kernel_params", None),
                n_centers=config.get("n_centers", 50),
                learn_centers=config.get("learn_centers", True),
            )

        elif name == "mlp":
            return Extensionlayer(  # <- dein residual-SiLU MLP
                input_size=input_size,
                output_size=output_size,
                hidden_size=config.get("hidden_size", 128),
                num_layers=config.get("num_layers", 10),
                dropout_rate=config.get("dropout_rate", 0.1),
            )

        else:
            raise ValueError(f"Unbekannter Extension-Typ: '{name}'")
        
    @staticmethod
    def get_extension_params(extension_module: nn.Module):
        """Gibt alle Parameter des Extension-Moduls zurück oder eine leere Liste, falls None."""
        if extension_module is None:
            return []
        return list(extension_module.parameters())
