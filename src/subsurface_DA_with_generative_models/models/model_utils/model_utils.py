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
    

class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'relu'
    ) -> None:
        super(ResNetBlock, self).__init__()

        self.activation = nn.ReLU()#get_activation_function(activation)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )


    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)

        residual = self.skip_conv(residual)

        x += residual
        return x


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
        mode: str = 'nearest',
        transposed_conv: bool = True
    ) -> None:
        super(ConvolutionalDownsampleLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding
        )
        self.downsample = nn.MaxPool2d(kernel_size=scale_factor)

    def forward(self, x):
        #x = self.downsample(x)
        x = self.conv(x)
        return x
    
class ResNetBlockWithDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        scale_factor: int = 2,
        mode: str = 'nearest',
        activation: str = 'relu'
    ) -> None:
        super(ResNetBlockWithDownsample, self).__init__()

        self.activation = get_activation_function(activation)

        self.resnet_block = ResNetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation
        )

        self.downsample = ConvolutionalDownsampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            scale_factor=scale_factor,
            mode=mode
        )


    def forward(self, x):
        
        x = self.resnet_block(x)
        x = self.activation(x)
        x = self.downsample(x)
        
        return x
    
def get_downsample_conv_layers(
    first_layer_channels: int,
    num_channels: list,
    resnet: bool = True,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1
    ):

    
    conv_layers = nn.ModuleList()

    if resnet:
        for i in range(0, len(num_channels)):
            if i == 0:
                conv_layers.append(
                    ResNetBlockWithDownsample(
                        in_channels=first_layer_channels,
                        out_channels=num_channels[i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )
            else:
                conv_layers.append(
                    ResNetBlockWithDownsample(
                        in_channels=num_channels[i-1],
                        out_channels=num_channels[i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                )
    else:
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
    


class ResNetBlockWithUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        scale_factor: int = 2,
        mode: str = 'nearest',
        activation: str = 'relu',
        transposed_conv: bool = False
    ) -> None:
        super(ResNetBlockWithUpsample, self).__init__()

        self.activation = get_activation_function(activation)

        self.resnet_block = ResNetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation
        )

        self.upsample = ConvolutionalUpsampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            scale_factor=scale_factor,
            mode=mode,
            transposed_conv = True
        )


    def forward(self, x):
        x = self.resnet_block(x)
        x = self.activation(x)
        x = self.upsample(x)
        
        return x

class ConvolutionalUpsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        scale_factor: int = 2,
        mode: str = 'nearest',
        transposed_conv: bool = True
    ) -> None:
        super(ConvolutionalUpsampleLayer, self).__init__()

        self.transposed_conv = transposed_conv

        if transposed_conv:
            self.upsample = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding
            )
        else:
            self.upsample = UpsampleLayer(
                scale_factor=scale_factor,
                mode=mode
            )
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )

    def forward(self, x):
        if self.transposed_conv:
            x = self.upsample(x)
        else:
            x = self.upsample(x)
            x = self.conv(x)
        return x

def get_upsample_conv_layers(
    first_layer_channels: int,
    num_channels: list,
    resnet: bool = True,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    transposed_conv: bool = True
    ):

    conv_layers = nn.ModuleList()

    if resnet:
        for i in range(0, len(num_channels)):
            if i == 0:
                conv_layers.append(
                    ResNetBlockWithUpsample(
                        in_channels=first_layer_channels,
                        out_channels=num_channels[i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1,
                        transposed_conv=transposed_conv
                    )
                )
            else:
                conv_layers.append(
                    ResNetBlockWithUpsample(
                        in_channels=num_channels[i-1],
                        out_channels=num_channels[i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        transposed_conv=transposed_conv
                    )
                )
    else:
        for i in range(0, len(num_channels)):
            if i == 0:
                conv_layers.append(
                    ConvolutionalUpsampleLayer(
                        in_channels=first_layer_channels,
                        out_channels=num_channels[i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        transposed_conv=transposed_conv
                    )
                )
            else:
                conv_layers.append(
                    ConvolutionalUpsampleLayer(
                        in_channels=num_channels[i-1],
                        out_channels=num_channels[i],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        transposed_conv=transposed_conv
                    )
                )
    
    return conv_layers