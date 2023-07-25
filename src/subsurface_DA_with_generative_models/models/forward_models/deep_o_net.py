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
    
class DeepONet(nn.Module):
    def __init__(
        self,
        num_dense_neurons: list,
        num_channels: int,
        activation: str = 'gelu'
    ) -> None:
        super(DeepONet, self).__init__()

        self.num_dense_neurons = num_dense_neurons
        self.num_channels = num_channels

        # reverse the order of the channels
        self.num_upsample_channels = [channels for channels in num_channels[::-1]]
        self.activation = model_utils.get_activation_function(activation)

        latent_height = int(64/(2**(len(num_channels)-1)))
        latent_width = int(64/(2**(len(num_channels)-1)))
        latent_channels = num_channels[-1]

        self.downsample_layers = model_utils.get_downsample_conv_layers(
            first_layer_channels = 2,
            num_channels = num_channels,
            resnet = False,
        )

        self.dynamic_input_encode_1 = nn.Linear(
            in_features=latent_channels,
            out_features=latent_channels
        )

        self.positional_encoding = PositionalEncoding(
            d_model=latent_channels
        )

        # Dynamic parameter encoder
        self.dynamic_parameter_encode_1 = nn.Conv1d(
            in_channels=1,
            out_channels=latent_height*latent_width,
            kernel_size=1,
        )
        self.dynamic_parameter_encode_2 = nn.Linear(
            in_features=61,
            out_features=latent_channels
        )
        self.dynamic_parameter_encode_3 = nn.Linear(
            in_features=latent_channels,
            out_features=latent_channels
        )
        self.dynamic_parameter_batch_norm = nn.BatchNorm1d(
            num_features=latent_height*latent_width
        )

        # Transformer layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=latent_channels,
                nhead=4,
                batch_first=True
            ),
            num_layers=3,
            norm=nn.LayerNorm(latent_channels),
        )

        self.dynamic_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=latent_channels,
                nhead=4,
                batch_first=True
            ),
            num_layers=3,
            norm=nn.LayerNorm(latent_channels),
        )


        self.final_3D_conv_layer_1 = nn.Conv3d(
            in_channels=num_channels[0],
            out_channels=2,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self.final_3D_conv_layer_2 = nn.Conv3d(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

    def prepare_spatial_data(
        self,
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None
    ):
        
        batch_size = static_spatial_parameters.shape[0]
        num_input_channels = static_spatial_parameters.shape[1]
        x_dim = static_spatial_parameters.shape[2]
        y_dim = static_spatial_parameters.shape[3]
        num_time_steps = dynamic_point_parameters.shape[-1]
        
        static_spatial_parameters = static_spatial_parameters.unsqueeze(dim=2)
        static_spatial_parameters = static_spatial_parameters.repeat(1, 1, num_time_steps, 1, 1)
        static_spatial_parameters = static_spatial_parameters.transpose(1, 2)
        static_spatial_parameters = static_spatial_parameters.reshape(
            batch_size*num_time_steps, num_input_channels, x_dim, y_dim
            )
        dynamic_spatial_parameters = dynamic_spatial_parameters.transpose(1, 2)
        dynamic_spatial_parameters = dynamic_spatial_parameters.reshape(
            batch_size*num_time_steps, dynamic_spatial_parameters.shape[2], x_dim, y_dim
            )

        spatial_parameters = torch.cat(
            (static_spatial_parameters, dynamic_spatial_parameters), 
            dim=1
            )

        spatial_parameters = static_spatial_parameters

        return spatial_parameters

    def dynamic_parameter_encoder(
        self,
        dynamic_point_parameters: torch.Tensor
    ):
        
        dynamic_point_parameters = self.activation(self.dynamic_parameter_encode_1(dynamic_point_parameters))
        dynamic_point_parameters = self.activation(self.dynamic_parameter_encode_2(dynamic_point_parameters))
        dynamic_point_parameters = self.activation(self.dynamic_parameter_encode_3(dynamic_point_parameters))
        dynamic_point_parameters = self.dynamic_parameter_batch_norm(dynamic_point_parameters)
        dynamic_point_parameters = self.positional_encoding(dynamic_point_parameters)

        dynamic_point_parameters = self.dynamic_transformer_encoder(
            src=dynamic_point_parameters
        )

        return dynamic_point_parameters

    def forward(
        self,
        static_point_parameters: torch.Tensor = None,
        static_spatial_parameters: torch.Tensor = None,
        dynamic_point_parameters: torch.Tensor = None,
        dynamic_spatial_parameters: torch.Tensor = None
    ):
        
        batch_size = static_spatial_parameters.shape[0]
        x_dim = static_spatial_parameters.shape[2]
        y_dim = static_spatial_parameters.shape[3]
        num_time_steps = dynamic_point_parameters.shape[-1]

        spatial_parameters = self.prepare_spatial_data(
            static_point_parameters=static_point_parameters,
            static_spatial_parameters=static_spatial_parameters,
            dynamic_point_parameters=dynamic_point_parameters,
            dynamic_spatial_parameters=dynamic_spatial_parameters
        ) # (batch_size*num_time_steps, num_input_channels, x_dim, y_dim)
        
        
        skip_connections, input_data = self.u_net_model.encoder(spatial_parameters)

        # (batch_size*num_time_steps, num_channels[-1], latent_height, latent_width)
        input_data = self.u_net_model.bottleneck1(input_data) 

        latent_height = input_data.shape[2]
        latent_width = input_data.shape[3]

        # (batch_size*num_time_steps, num_channels[-1], latent_height*latent_width)
        input_data = input_data.view(-1, self.num_channels[-1], latent_height * latent_width)
        input_data = input_data.transpose(1, 2)

        # (batch_size*num_time_steps, num_channels[-1], latent_height*latent_width)
        input_data = self.dynamic_input_encode_1(input_data)
        input_data = self.positional_encoding(input_data)

        # (batch_size, latent_height*latent_width, num_channels[-1])
        dynamic_point_parameters = self.dynamic_parameter_encoder(dynamic_point_parameters)

        # (batch_size*num_time_steps, latent_height*latent_width, num_channels[-1])
        dynamic_point_parameters = torch.tile(dynamic_point_parameters, (num_time_steps, 1, 1))

        input_data = self.transformer_decoder(
            tgt=input_data,
            memory=dynamic_point_parameters
        )
        input_data = input_data.transpose(1, 2)

        # (batch_size*num_time_steps, num_channels[-1], latent_height, latent_width)
        input_data = input_data.view(-1, self.num_channels[-1], latent_height, latent_width)

        # (batch_size*num_time_steps, num_channels[-1], latent_height, latent_width)
        input_data = self.u_net_model.bottleneck2(input_data)

        # (batch_size*num_time_steps, num_channels[0], x_dim, y_dim)
        input_data = self.u_net_model.decoder(input_data, skip_connections)

        # (batch_size*num_time_steps, 2, x_dim, y_dim)
        #input_data = self.final_conv_layer(input_data)

        # (batch_size, 2, num_time_steps, x_dim, y_dim)
        input_data = input_data.view(batch_size, self.num_channels[0], num_time_steps, x_dim, y_dim)

        # (batch_size, 2, num_time_steps, x_dim, y_dim)
        input_data = self.final_3D_conv_layer_1(input_data)
        input_data = self.activation(input_data)
        input_data = self.final_3D_conv_layer_2(input_data)

        return input_data