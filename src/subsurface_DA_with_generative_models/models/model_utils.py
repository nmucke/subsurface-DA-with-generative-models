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

def get_conv_layers(
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
                nn.Conv2d(
                    in_channels=first_layer_channels,
                    out_channels=num_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
        else:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=num_channels[i-1],
                    out_channels=num_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
    
    return conv_layers

def get_transposed_conv_layers(
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
                nn.ConvTranspose2d(
                    in_channels=first_layer_channels,
                    out_channels=num_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
        else:
            conv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=num_channels[i-1],
                    out_channels=num_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
    
    return conv_layers