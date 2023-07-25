import pdb
import torch
import torch.nn as nn

import subsurface_DA_with_generative_models.models.model_utils.model_utils as model_utils
from subsurface_DA_with_generative_models.models.model_utils.u_net_layers import UnetModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ParameterGAN(nn.Module):
    def __init__(
        self,
        generator_args: dict,
        critic_args: dict,
        device: str = 'cpu'
    ) -> None:
        
        super(ParameterGAN, self).__init__()

        self.generator = ParameterGenerator(**generator_args)
        self.critic =  ParameterCritic(**critic_args)

        self.latent_dim = self.generator.latent_dim

    @property
    def device(self):
        return next(self.parameters()).device.type
    
    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor
    ):
        pass


class ParameterGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_dense_neurons: list,
        num_channels: int,
        activation: str = 'gelu',
        transposed_conv: bool = False,
        resnet: bool = False,
    ) -> None:
        super(ParameterGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_dense_neurons = num_dense_neurons
        self.num_channels = num_channels

        self.sigmoid = nn.Sigmoid()

        # reverse the order of the channels
        self.num_upsample_channels = [channels for channels in num_channels[::-1]]
        self.activation = model_utils.get_activation_function(activation)

        self.dense_layers = model_utils.get_dense_layers(
            first_layer_neurons=self.latent_dim,
            num_dense_neurons=self.num_dense_neurons
        )

        self.final_dense_layer = nn.Linear(
            in_features=self.num_dense_neurons[-1],
            out_features=num_channels[0]*2*2
        )

        self.upsample_conv_layers = model_utils.get_upsample_conv_layers(
            first_layer_channels=self.num_channels[0],# * 2,
            num_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            resnet=resnet,
            transposed_conv=transposed_conv
        )

        self.final_res_layer = model_utils.ConvolutionalUpsampleLayer(
            in_channels=num_channels[-1],
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            transposed_conv=transposed_conv
        )
        self.final_conv_layer = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
    
    def forward(
            self, 
            latent: torch.Tensor,
        ):
        
        z = latent
        for dense_layer in self.dense_layers:
            z = self.activation(dense_layer(z))
        
        z = self.activation(self.final_dense_layer(z))

        z = z.view(-1, self.num_channels[0], 2, 2)
        for upsample_conv_layer in self.upsample_conv_layers:
            z = self.activation(upsample_conv_layer(z))

        z = self.activation(self.final_res_layer(z))
        z = self.final_conv_layer(z)
        z = z[:, :, 0:64, 0:64]

        return z

class ParameterCritic(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_dense_neurons: list,
        activation: str = 'gelu',
        resnet: bool = False,
        wasserstein: bool = False,
    ) -> None:
        super(ParameterCritic, self).__init__()

        self.num_channels = num_channels

        self.wasserstein = wasserstein
        self.sigmoid = nn.Sigmoid()

        first_layer_channels = 2
        
        self.num_dense_neurons = num_dense_neurons
        self.activation = model_utils.get_activation_function(activation)

        self.conv_layers = model_utils.get_downsample_conv_layers(
            first_layer_channels=first_layer_channels,
            num_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            resnet=resnet
        )

        self.dense_layers = model_utils.get_dense_layers(
            first_layer_neurons=4*num_channels[-1],
            num_dense_neurons=self.num_dense_neurons
        )

        self.output_layer = nn.Linear(
            in_features=self.num_dense_neurons[-1],
            out_features=1
        )
    
    def forward(
        self, 
        static_spatial_parameters: torch.Tensor = None,
        ):

        for conv_layer in self.conv_layers:
            static_spatial_parameters = self.activation(conv_layer(static_spatial_parameters))
        
        static_spatial_parameters = static_spatial_parameters.flatten(start_dim=1)
        
        for dense_layer in self.dense_layers:
            static_spatial_parameters = self.activation(dense_layer(static_spatial_parameters))

        static_spatial_parameters = self.output_layer(static_spatial_parameters)
        
        if self.wasserstein:
            return static_spatial_parameters
        else:
            return self.sigmoid(static_spatial_parameters)
    