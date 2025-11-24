import torch
import torch.nn as nn
from src.layers.batchnorm_relu import BatchNormReLU

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_relu = BatchNormReLU(out_channels)
    
    def forward(self, x):
        x = self.pointwise(x)
        x = self.bn_relu(x)
        return x
