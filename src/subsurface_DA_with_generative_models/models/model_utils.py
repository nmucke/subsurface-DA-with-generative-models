import pdb
import torch
import torch.nn as nn

def get_activation_function(activation_func):

    if activation_func == 'relu':
        return nn.ReLU()
    elif activation_func == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation_func == 'tanh':
        return nn.Tanh()
    elif activation_func == 'sigmoid':
        return nn.Sigmoid()
    elif activation_func == 'softmax':
        return nn.Softmax(dim=1)
    elif activation_func == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError
    
def get_dense_layers(
    first_layer_neurons: list, 
    num_dense_neurons: list
    ):

    dense_layers = nn.ModuleList()
    for i in range(len(num_dense_neurons)):
        if i == 0:
            dense_layers.append(
                nn.Linear(
                    first_layer_neurons,
                    num_dense_neurons[i]
                )
            )
        else:
            dense_layers.append(
                nn.Linear(
                    num_dense_neurons[i-1],
                    num_dense_neurons[i]
                )
            )

    return dense_layers

class ConvolutionalDownsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        scale_factor: int = 2,
        mode: str = 'nearest'
    ) -> None:
        super(ConvolutionalDownsampleLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.downsample = nn.MaxPool2d(kernel_size=scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.downsample(x)
        return x
    
def get_downsample_conv_layers(
    first_layer_channels: int,
    num_channels: list,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1
    ):

    
    conv_layers = nn.ModuleList()
    for i in range(0, len(num_channels)):
        if i == 0:
            conv_layers.append(
                ConvolutionalDownsampleLayer(
                    in_channels=first_layer_channels,
                    out_channels=num_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
        else:
            conv_layers.append(
                ConvolutionalDownsampleLayer(
                    in_channels=num_channels[i-1],
                    out_channels=num_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
    
    return conv_layers

class UpsampleLayer(nn.Module):
    def __init__(
        self,
        scale_factor: int = 2,
        mode: str = 'nearest'
    ) -> None:
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
    



class ConvolutionalUpsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        scale_factor: int = 2,
        mode: str = 'nearest'
    ) -> None:
        super(ConvolutionalUpsampleLayer, self).__init__()

        self.upsample = UpsampleLayer(scale_factor=scale_factor, mode=mode)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

def get_upsample_conv_layers(
    first_layer_channels: int,
    num_channels: list,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1
    ):

    conv_layers = nn.ModuleList()
    for i in range(0, len(num_channels)):
        if i == 0:
            conv_layers.append(
                ConvolutionalUpsampleLayer(
                    in_channels=first_layer_channels,
                    out_channels=num_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                )
            )
        else:
            conv_layers.append(
                ConvolutionalUpsampleLayer(
                    in_channels=num_channels[i-1],
                    out_channels=num_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
    
    return conv_layers