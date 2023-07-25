import pdb
import torch
import torch.nn as nn

from subsurface_DA_with_generative_models.models.model_utils.model_utils import get_activation_function


class conv_block(nn.Module):
    def __init__(
        self, 
        in_c, 
        out_c, 
        activation='relu',
        residual=True,
        ):
        super().__init__()

        self.residual = residual

        self.activation = get_activation_function(activation)

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        if self.residual:
            self.skip_conv = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):

        residual = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.residual:
            residual = self.skip_conv(residual)
            x += residual

        x = self.activation(x)

        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, activation='relu'):
        super().__init__()

        self.conv = conv_block(in_c, out_c, activation=activation)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):

        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, activation='relu'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c, activation=activation)

    def forward(self, inputs, skip):

        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UnetModel(nn.Module):
    def __init__(self, first_layer_channels, num_channels, activation='relu'):
        super().__init__()

        """ Encoder """
        self.encoder_blocks = nn.ModuleList()
        for i in range(0, len(num_channels)-1):
            if i == 0:
                self.encoder_blocks.append(
                    encoder_block(
                        in_c=first_layer_channels,
                        out_c=num_channels[i],
                        activation=activation
                    )
                )
            else:
                self.encoder_blocks.append(
                    encoder_block(
                        in_c=num_channels[i-1],
                        out_c=num_channels[i],
                        activation=activation
                    )
                )
            

        """ Bottleneck """
        self.bottleneck1 = conv_block(num_channels[-2], num_channels[-1], activation=activation)
        self.bottleneck2 = conv_block(num_channels[-1], num_channels[-1], activation=activation)

        self.decoder_block = nn.ModuleList()
        for i in range(0, len(num_channels)-1):
            self.decoder_block.append(
                decoder_block(
                    in_c=num_channels[-1-i],
                    out_c=num_channels[-2-i],
                    activation=activation
                )
            )

    def encoder(self, inputs):
            
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            s, inputs = encoder_block(inputs)
            skip_connections.append(s)
        return skip_connections, inputs
    
    def decoder(self, inputs, skip_connections):
        for i, decoder_block in enumerate(self.decoder_block):
            inputs = decoder_block(inputs, skip_connections[-1-i])
        return inputs
            

    def forward(self, inputs):


        """ Encoder """
        skip_connections, inputs = self.encoder(inputs)
        
        """ Bottleneck """
        inputs = self.bottleneck1(inputs)

        """ Decoder """
        inputs = self.decoder(inputs, skip_connections)
            
        return inputs