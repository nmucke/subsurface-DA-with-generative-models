import pdb
import torch
import torch.nn as nn

import subsurface_DA_with_generative_models.models.model_utils as model_utils
from subsurface_DA_with_generative_models.models.u_net_layers import UnetModel

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

class GAN(nn.Module):
    def __init__(
        self,
        generator_args: dict,
        critic_args: dict,
        device: str = 'cpu'
    ) -> None:
        
        super(GAN, self).__init__()

        self.generator = UnetGenerator(**generator_args)
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

class UnetGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_dense_neurons: list,
        num_channels: int,
        activation: str = 'gelu'
    ) -> None:
        super(UnetGenerator, self).__init__()

        self.latent_dim = latent_dim
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
            latent_samples: torch.Tensor = None,
            input_data: torch.Tensor = None,
            dynamic_input_data: torch.Tensor = None
        ):

        skip_connections, input_data = self.u_net_model.encoder(input_data)
        input_data = self.u_net_model.bottleneck1(input_data)

        if dynamic_input_data is not None:
            latent_height = input_data.shape[2]
            latent_width = input_data.shape[3]

            input_data = input_data.view(-1, self.num_channels[-1], latent_height * latent_width)
            input_data = input_data.transpose(1, 2)
            input_data = self.positional_encoding(input_data)

            dynamic_input_data = self.activation(self.dynamic_encode_1(dynamic_input_data))
            dynamic_input_data = self.activation(self.dynamic_encode_2(dynamic_input_data))
            dynamic_input_data = self.dynamic_batch_norm(dynamic_input_data)
            dynamic_input_data = self.positional_encoding(dynamic_input_data)

            dynamic_input_data = self.dynamic_transformer_encoder(
                src=dynamic_input_data
            )

            input_data = self.transformer_decoder(
                tgt=input_data,
                memory=dynamic_input_data
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
            output_data: torch.Tensor,
            input_data: torch.Tensor = None,
            dynamic_input_data: torch.Tensor = None
        ):
        

        data = torch.cat((output_data, input_data), dim=1)

        for conv_layer in self.conv_layers:
            data = self.activation(conv_layer(data))
        
        latent_height = data.shape[2]
        latent_width = data.shape[3]

        data = data.view(-1, self.num_channels[-1], latent_height * latent_width)
        data = data.transpose(1, 2)
        data = self.positional_encoding(data)

        if dynamic_input_data is not None:
            
            dynamic_input_data = self.activation(self.dynamic_encode_1(dynamic_input_data))
            dynamic_input_data = self.activation(self.dynamic_encode_2(dynamic_input_data))
            dynamic_input_data = self.dynamic_batch_norm(dynamic_input_data)
            dynamic_input_data = self.positional_encoding(dynamic_input_data)

            dynamic_input_data = self.dynamic_transformer_encoder(
                src=dynamic_input_data
            )

            data = self.transformer_decoder(
                tgt=data,
                memory=dynamic_input_data
            )

        data = data.flatten(start_dim=1)

        
        for dense_layer in self.dense_layers:
            data = self.activation(dense_layer(data))

        data = self.output_layer(data)

        
        return data
    