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

class UNetGAN(nn.Module):
    def __init__(
        self,
        generator_args: dict,
        critic_args: dict,
        device: str = 'cpu'
    ) -> None:
        
        super(UNetGAN, self).__init__()

        self.generator = UnetGenerator(**generator_args)
        self.critic =  Critic(**critic_args)

    @property
    def device(self):
        return next(self.parameters()).device.type
    
    def forward(
        self,
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None
    ):
        return self.generator(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters
        )


class UnetGenerator(nn.Module):
    def __init__(
        self,
        num_dense_neurons: list,
        num_channels: int,
        activation: str = 'gelu'
    ) -> None:
        super(UnetGenerator, self).__init__()

        self.num_dense_neurons = num_dense_neurons
        self.num_channels = num_channels

        # reverse the order of the channels
        self.num_upsample_channels = [channels for channels in num_channels[::-1]]
        self.activation = model_utils.get_activation_function(activation)

        self.u_net_model = UnetModel(
            first_layer_channels=3,
            num_channels=num_channels
        )
        self.dynamic_encode_1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=1,
        )
        self.dynamic_encode_2 = nn.Linear(
            in_features=61,
            out_features=self.num_channels[-1]
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.num_channels[-1],
                nhead=8,
                batch_first=True
            ),
            num_layers=3,
            norm=nn.LayerNorm(self.num_channels[-1]),
        )

        self.dynamic_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.num_channels[-1],
                nhead=8,
                batch_first=True
            ),
            num_layers=3,
            norm=nn.LayerNorm(self.num_channels[-1]),
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.num_channels[-1]
        )
        self.dynamic_batch_norm = nn.BatchNorm1d(
            num_features=16
        )

        self.final_conv_layer = nn.Conv2d(
            in_channels=num_channels[0],
            out_channels=2,
            kernel_size=1,
            stride=1,
        )

    def forward(
        self,
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None
    ):
        
        
        num_time_steps = dynamic_point_parameters.shape[-1]
        
        static_spatial_parameters = static_spatial_parameters.unsqueeze(dim=2)
        static_spatial_parameters = static_spatial_parameters.repeat(1, 1, num_time_steps, 1, 1)
        static_spatial_parameters = static_spatial_parameters.transpose(1, 2)
        static_spatial_parameters = static_spatial_parameters.reshape(
            -1, static_spatial_parameters.shape[2], static_spatial_parameters.shape[3], static_spatial_parameters.shape[4]
            )
        dynamic_spatial_parameters = dynamic_spatial_parameters.transpose(1, 2)
        dynamic_spatial_parameters = dynamic_spatial_parameters.reshape(
            -1, dynamic_spatial_parameters.shape[2], dynamic_spatial_parameters.shape[3], dynamic_spatial_parameters.shape[4]
            )
        
        dynamic_point_parameters = torch.tile(dynamic_point_parameters, (num_time_steps, 1, 1))

        spatial_parameters = torch.cat(
            (static_spatial_parameters, dynamic_spatial_parameters), 
            dim=1
            )
        
        skip_connections, input_data = self.u_net_model.encoder(spatial_parameters)
        input_data = self.u_net_model.bottleneck1(input_data)

        if dynamic_point_parameters is not None:
            latent_height = input_data.shape[2]
            latent_width = input_data.shape[3]

            input_data = input_data.view(-1, self.num_channels[-1], latent_height * latent_width)
            input_data = input_data.transpose(1, 2)
            input_data = self.positional_encoding(input_data)

            dynamic_point_parameters = self.activation(self.dynamic_encode_1(dynamic_point_parameters))
            dynamic_point_parameters = self.activation(self.dynamic_encode_2(dynamic_point_parameters))
            dynamic_point_parameters = self.dynamic_batch_norm(dynamic_point_parameters)
            dynamic_point_parameters = self.positional_encoding(dynamic_point_parameters)

            dynamic_point_parameters = self.dynamic_transformer_encoder(
                src=dynamic_point_parameters
            )

            input_data = self.transformer_decoder(
                tgt=input_data,
                memory=dynamic_point_parameters
            )

            input_data = input_data.transpose(1, 2)
            input_data = input_data.view(-1, self.num_channels[-1], latent_height, latent_width)
        
            input_data = self.u_net_model.bottleneck2(input_data)

        input_data = self.u_net_model.decoder(input_data, skip_connections)
        
        return self.final_conv_layer(input_data)




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
            first_layer_channels=self.num_channels[0],# * 2,
            num_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.downsample_conv_layers = model_utils.get_downsample_conv_layers(
            first_layer_channels=3,
            num_channels=self.num_upsample_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.final_conv_layer = nn.Conv2d(
            in_channels=num_channels[-1],
            out_channels=2,
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

        #z = torch.cat((z, input_data), dim=1)
        z = input_data

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

        self.num_channels = num_channels

        first_layer_channels = 5
        
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
            first_layer_neurons=2048,
            num_dense_neurons=self.num_dense_neurons
        )

        self.output_layer = nn.Linear(
            in_features=self.num_dense_neurons[-1],
            out_features=1
        )

        self.dynamic_encode_1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=1,
        )
        self.dynamic_encode_2 = nn.Linear(
            in_features=61,
            out_features=self.num_channels[-1]
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.num_channels[-1],
                nhead=8,
                batch_first=True
            ),
            num_layers=3,
            norm=nn.LayerNorm(self.num_channels[-1]),
        )


        self.dynamic_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.num_channels[-1],
                nhead=8,
                batch_first=True
            ),
            num_layers=3,
            norm=nn.LayerNorm(self.num_channels[-1]),
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.num_channels[-1]
        )

        self.dynamic_batch_norm = nn.BatchNorm1d(
            num_features=16
        )

    
    def forward(
        self,
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None,
        output_variables: torch.Tensor = None
    ):
        
        num_time_steps = dynamic_point_parameters.shape[-1]

        output_variables = output_variables.transpose(1, 2)
        output_variables = output_variables.reshape(
            -1, output_variables.shape[2], output_variables.shape[3], output_variables.shape[4]
            )
        static_spatial_parameters = static_spatial_parameters.unsqueeze(dim=2)
        static_spatial_parameters = static_spatial_parameters.repeat(1, 1, num_time_steps, 1, 1)
        static_spatial_parameters = static_spatial_parameters.transpose(1, 2)
        static_spatial_parameters = static_spatial_parameters.reshape(
            -1, static_spatial_parameters.shape[2], static_spatial_parameters.shape[3], static_spatial_parameters.shape[4]
            )
        dynamic_spatial_parameters = dynamic_spatial_parameters.transpose(1, 2)
        dynamic_spatial_parameters = dynamic_spatial_parameters.reshape(
            -1, dynamic_spatial_parameters.shape[2], dynamic_spatial_parameters.shape[3], dynamic_spatial_parameters.shape[4]
            )
        
        dynamic_point_parameters = torch.tile(dynamic_point_parameters, (num_time_steps, 1, 1))
        
        data = torch.cat(
            (output_variables, static_spatial_parameters, dynamic_spatial_parameters), 
            dim=1
            )

        for conv_layer in self.conv_layers:
            data = self.activation(conv_layer(data))
        
        latent_height = data.shape[2]
        latent_width = data.shape[3]

        data = data.view(-1, self.num_channels[-1], latent_height * latent_width)
        data = data.transpose(1, 2)
        data = self.positional_encoding(data)


        if dynamic_point_parameters is not None:
            
            dynamic_point_parameters = self.activation(self.dynamic_encode_1(dynamic_point_parameters))
            dynamic_point_parameters = self.activation(self.dynamic_encode_2(dynamic_point_parameters))
            dynamic_point_parameters = self.dynamic_batch_norm(dynamic_point_parameters)
            dynamic_point_parameters = self.positional_encoding(dynamic_point_parameters)

            dynamic_point_parameters = self.dynamic_transformer_encoder(
                src=dynamic_point_parameters
            )

            data = self.transformer_decoder(
                tgt=data,
                memory=dynamic_point_parameters
            )

        data = data.flatten(start_dim=1)

        
        for dense_layer in self.dense_layers:
            data = self.activation(dense_layer(data))

        data = self.output_layer(data)
        return data
    