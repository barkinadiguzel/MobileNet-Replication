import torch
import torch.nn as nn

class AvgPoolLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(AvgPoolLayer, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        x = self.pool(x)
        return x
