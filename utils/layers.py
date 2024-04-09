import torch
import torch.nn as nn
from utils.utils import compute_output_shape, compute_output_padding, invalid_shape, compute_transpose_output_shape

from utils.errors import InvalidArchitectureError

def encoder(architecture, input_shape):
    in_channels = input_shape[0] 
    current_shape = (input_shape[1], input_shape[2])

    layers_shape = [current_shape]

    model = []

    for layer in range(architecture['conv_layers']):
        
        out_channels = architecture["conv_channels"][layer]
        kernel_size = architecture["conv_kernel_sizes"][layer]
        stride = architecture["conv_strides"][layer]
        padding = architecture["conv_paddings"][layer]
        
        sequential_nn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(negative_slope=0.15),
            nn.BatchNorm2d(out_channels)
        )
        
        model.append(sequential_nn)
        
        current_shape = compute_output_shape(current_shape, kernel_size, stride, padding)
        layers_shape.append(current_shape)
        
        if invalid_shape(current_shape):
            raise InvalidArchitectureError(shape=current_shape, layer=layer+1)
        
        in_channels = out_channels
        
    return nn.Sequential(*model), layers_shape
        
def decoder(architecture, encoder_shapes):
    in_channels = architecture["conv_channels"][-1]
    
    model = []
    
    for layer in range(architecture["conv_layers"] - 1, -1, -1):

        out_channels = architecture["conv_channels"][layer]
        kernel_size = architecture["conv_kernel_sizes"][layer]
        stride = architecture["conv_strides"][layer]
        padding = architecture["conv_paddings"][layer]
        
        current_shape = encoder_shapes[layer + 1]
        target_shape = encoder_shapes[layer]
        
        output_shape = compute_transpose_output_shape(current_shape, kernel_size, stride, padding)
        
        output_padding = compute_output_padding(output_shape, target_shape)
        
        sequential_nn = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding),
            nn.LeakyReLU(negative_slope=0.15),
            nn.BatchNorm2d(out_channels)
        )
        
        model.append(sequential_nn)
        
        in_channels = out_channels
        
    return nn.Sequential(*model)

def out_layer(architecture, input_shape):
    
    in_channels = architecture["conv_channels"][0]
    kernel_size = architecture["conv_kernel_sizes"][0]
    stride = architecture["conv_strides"][0]
    padding = architecture["conv_paddings"][0]

    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=in_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                         nn.SELU(),
                         nn.BatchNorm2d(in_channels),
                         nn.Conv2d(in_channels=in_channels,
                                   out_channels=input_shape[0],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                         nn.Sigmoid())