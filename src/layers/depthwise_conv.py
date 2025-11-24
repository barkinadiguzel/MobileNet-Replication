import torch
import torch.nn as nn
from src.layers.batchnorm_relu import BatchNormReLU

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn_relu = BatchNormReLU(in_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_relu(x)
        return x
