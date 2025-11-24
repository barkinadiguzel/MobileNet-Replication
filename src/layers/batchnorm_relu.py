import torch
import torch.nn as nn

class BatchNormReLU(nn.Module):
    def __init__(self, num_features):
        super(BatchNormReLU, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x
