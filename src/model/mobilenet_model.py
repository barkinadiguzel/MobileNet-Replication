import torch
import torch.nn as nn
from src.layers.depthwise_conv import DepthwiseConv
from src.layers.pointwise_conv import PointwiseConv
from src.layers.global_avgpool import GlobalAvgPool
from src.layers.flatten_layer import FlattenLayer

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileNetBlock, self).__init__()
        self.depthwise = DepthwiseConv(in_channels, stride=stride)
        self.pointwise = PointwiseConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_relu1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # MobileNet blocks (simplified example)
        self.layer1 = MobileNetBlock(32, 64, stride=1)
        self.layer2 = MobileNetBlock(64, 128, stride=2)
        self.layer3 = MobileNetBlock(128, 128, stride=1)
        self.layer4 = MobileNetBlock(128, 256, stride=2)
        self.layer5 = MobileNetBlock(256, 256, stride=1)
        self.layer6 = MobileNetBlock(256, 512, stride=2)

        # Repeat some blocks (like original MobileNet)
        self.layer7 = nn.Sequential(
            *[MobileNetBlock(512, 512, stride=1) for _ in range(5)]
        )

        self.layer8 = MobileNetBlock(512, 1024, stride=2)
        self.layer9 = MobileNetBlock(1024, 1024, stride=1)

        # Final pooling + classifier
        self.global_pool = GlobalAvgPool()
        self.flatten = FlattenLayer()
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
