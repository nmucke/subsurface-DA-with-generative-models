import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class LatentToOutputModel(nn.Module):
    def __init__(
        self,
        parameter_model: nn.Module,
        forward_model: nn.Module,
    ) -> None:
        
        super().__init__()
        
        self.parameter_model = parameter_model
        self.forward_model = forward_model

    def forward(
        self,
        static_spatial_parameters: torch.Tensor = None,
        static_point_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None,
    ) -> torch.Tensor:
        
        # Get latent
        static_spatial_parameters = self.parameter_model(latent=static_spatial_parameters,)
        
        # Get sample
        output = self.forward_model(
            static_spatial_parameters=static_spatial_parameters,
            static_point_parameters=static_point_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters,
        )
        
        return output
        
