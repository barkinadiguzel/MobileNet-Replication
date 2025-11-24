import torch
import torch.nn as nn

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) 
    
    def forward(self, x):
        x = self.pool(x)  
        x = torch.flatten(x, 1) 
        return x
