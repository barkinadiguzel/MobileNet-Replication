import torch
import torch.nn as nn

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    
    def forward(self, x):
        return torch.flatten(x, 1)  
