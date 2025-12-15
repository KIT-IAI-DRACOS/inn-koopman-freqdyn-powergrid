import torch
from .freia_allinone import FreiaINN
from ._glow import FreiaGlowINN
from ._i_ResNet import iResNetFreia
from ._ffjord import FFJORD
from ._hint import HintINN
from ._nice import FreiaNiceINN
from ._real_nvp import FreiaRealNVPINN
from ._id import IdentityINN
from pykoopman.regression._extension_factory import ExtensionFactory
from ._hybrid_inn_wrapper import HybridINNWrapper

class INNFactory:
    """
    Factory class for creating different INN types.
    """
    @staticmethod
    def instantiate_base_inn(config_inn):
        """
        Factory method for creating an INN model.
        
        Parameters
        ----------
        inn_type : str
            Type of the INN 
        config_inn : dict
            Configuration dictionary containing all necessary parameters for the model.
        
        Returns
        -------
        nn.Module
            An instance of the specified INN model class.
        """
        input_size = config_inn.get("input_size")
        hidden_size = config_inn.get("hidden_size")
        num_layers = config_inn.get("num_layers")
        init_identity = config_inn.get("init_identity", True)
        inn_type = config_inn.get("inn_type")
        
        if inn_type == "freiaallinone":
            inn_model = FreiaINN(input_size, hidden_size, num_layers, init_identity)
        
        elif inn_type == "iresnet":
            inn_model = iResNetFreia(input_size,hidden_size,num_layers)

        elif inn_type == "ffjord":
            inn_model = FFJORD(input_size,hidden_size)

        elif inn_type == "hint":
            inn_model = HintINN(input_size, hidden_size, num_layers)

        elif inn_type == "glow":
            inn_model = FreiaGlowINN(input_size, hidden_size, num_layers)

        elif inn_type == "nice":
            inn_model = FreiaNiceINN(input_size,hidden_size, num_layers)

        elif inn_type == "realnvp":
            inn_model = FreiaRealNVPINN(input_size, hidden_size, num_layers)
        
        elif inn_type == "id":
            inn_model = IdentityINN(input_size)

        else:
            raise ValueError(f"Unknown INN type: {inn_type}")
                
        return inn_model
    
    @staticmethod
    def create_inn(config_inn):
        base = INNFactory.instantiate_base_inn(config_inn)
        ext_cfg = config_inn.get("extension_config")
        if ext_cfg and ext_cfg.get("extension_output_size", 0) > 0:
            ext = ExtensionFactory.create_extension(
                name=ext_cfg["extension_type"],
                input_size=config_inn["input_size"],
                output_size=ext_cfg["extension_output_size"],
                config=ext_cfg
            )
            return HybridINNWrapper(base, ext)
        else:
            return base
        
    @staticmethod
    def get_extension_params(inn_module):
        params = list(inn_module.parameters())
        
        # Prüfen, ob das inn_module eine Extension besitzt (HybridINNWrapper)
        if hasattr(inn_module, "extension") and inn_module.extension is not None:
            ext_params = ExtensionFactory.get_extension_params(inn_module.extension)
            params.extend(ext_params)
        
        return params



