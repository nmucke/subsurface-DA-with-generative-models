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
        self.activation = model_utils.get_activation_function(activation)

        self.dense_layers = model_utils.get_dense_layers(
            first_layer_neurons=self.latent_dim,
            num_dense_neurons=self.num_dense_neurons
        )

        self.trans_conv_layers = model_utils.get_transposed_conv_layers(
            first_layer_channels=self.num_dense_neurons[-1]//4,
            num_channels=num_channels,
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

        z = z.view(-1, self.num_dense_neurons[-1]//4, 2, 2)

        for trans_conv_layer in self.trans_conv_layers:
            z = self.activation(trans_conv_layer(z))
            pdb.set_trace()
               
        return z

class Critic(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_dense_neurons: list,
        activation: str = 'gelu'
    ) -> None:
        super(Critic, self).__init__()

        first_layer_channels = 1
        
        self.num_dense_neurons = num_dense_neurons
        self.activation = model_utils.get_activation_function(activation)

        self.conv_layers = model_utils.get_conv_layers(
            first_layer_channels=first_layer_channels,
            num_channels=num_channels,
            kernel_size=3,
            stride=2,
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

        for conv_layer in self.conv_layers:
            output_data = self.activation(conv_layer(output_data))

        output_data = output_data.flatten(start_dim=1)

        for dense_layer in self.dense_layers:
            output_data = self.activation(dense_layer(output_data))

        output_data = self.output_layer(output_data)
               
        return output_data
    