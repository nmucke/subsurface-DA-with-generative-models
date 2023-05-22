import pdb
import torch
import torch.nn as nn

import subsurface_DA_with_generative_models.models.model_utils as model_utils


class GAN(nn.Module):
    def __init__(
        self,
        generator_args: dict,
        critic_args: dict,
        device: str = 'cpu'
    ) -> None:
        
        super(GAN, self).__init__()

        self.generator = Generator(**generator_args)
        self.critic =  Critic(**critic_args)

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


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_dense_neurons: list,
        num_channels: int,
        activation: str = 'gelu'
    ) -> None:
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_dense_neurons = num_dense_neurons
        self.num_channels = num_channels

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
            first_layer_channels=self.num_channels[0] * 2,
            num_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.downsample_conv_layers = model_utils.get_downsample_conv_layers(
            first_layer_channels=6,
            num_channels=self.num_upsample_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.final_conv_layer = nn.Conv2d(
            in_channels=num_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )
    
    def forward(
            self, 
            latent_samples: torch.Tensor,
            input_data: torch.Tensor = None
        ):
        
        
        z = latent_samples
        for dense_layer in self.dense_layers:
            z = self.activation(dense_layer(z))
        
        z = self.activation(self.final_dense_layer(z))

        z = z.view(-1, self.num_channels[0], 2, 2)

        for downsample_conv_layer in self.downsample_conv_layers:
            input_data = self.activation(downsample_conv_layer(input_data))

        z = torch.cat((z, input_data), dim=1)

        for upsample_conv_layer in self.upsample_conv_layers:
            z = self.activation(upsample_conv_layer(z))
        
        z = self.final_conv_layer(z)
               
        return z

class Critic(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_dense_neurons: list,
        activation: str = 'gelu'
    ) -> None:
        super(Critic, self).__init__()

        first_layer_channels = 7
        
        self.num_dense_neurons = num_dense_neurons
        self.activation = model_utils.get_activation_function(activation)

        self.conv_layers = model_utils.get_downsample_conv_layers(
            first_layer_channels=first_layer_channels,
            num_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.dense_layers = model_utils.get_dense_layers(
            first_layer_neurons=128,
            num_dense_neurons=self.num_dense_neurons
        )

        self.output_layer = nn.Linear(
            in_features=self.num_dense_neurons[-1],
            out_features=1
        )
    
    def forward(
            self, 
            output_data: torch.Tensor,
            input_data: torch.Tensor = None
        ):

        data = torch.cat((output_data, input_data), dim=1)

        for conv_layer in self.conv_layers:
            data = self.activation(conv_layer(data))

        data = data.flatten(start_dim=1)

        for dense_layer in self.dense_layers:
            data = self.activation(dense_layer(data))

        data = self.output_layer(data)
        
        return data
    