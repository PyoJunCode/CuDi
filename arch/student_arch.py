import torch
import torch.nn as nn

from arch.layers import *

class StudentNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, stride=1, groups=3),
            ConvBlock(3, 16, 1, activation="ReLU"),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=1, groups=16),
            ConvBlock(16, 16, 1, activation="ReLU"),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=1, groups=16),
            ConvBlock(16, 16, 1, activation="ReLU"),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, stride=1, groups=16),
            ConvBlock(16, 16, 1, activation="ReLU"),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32),
            ConvBlock(32, 16, 1, activation="ReLU"),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32),
            ConvBlock(32, 16, 1, activation="ReLU"),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32),
            nn.Conv2d(32, 6, 1, stride=1)
        )
    
    def _init_weight(self, block):
        if isinstance(block, nn.Conv2d):
            block.weight.data.normal_(mean=0., std=0.02)

    def init_weight(self):
        self.conv1.apply(self._init_weight)
        self.conv2.apply(self._init_weight)
        self.conv3.apply(self._init_weight)
        self.conv4.apply(self._init_weight)
        self.conv5.apply(self._init_weight)
        self.conv6.apply(self._init_weight)
        self.conv7.apply(self._init_weight)
    
    def forward(self, x):

        down_x = F.interpolate(x, scale_factor=0.25, mode="bilinear")

        conv1_out = self.conv1(down_x)

        conv2_out = self.conv2(conv1_out)

        conv3_out = self.conv3(conv2_out)

        conv4_out = self.conv4(conv3_out)

        conv5_out = self.conv5(torch.cat((conv4_out, conv3_out), 1))

        conv6_out = self.conv6(torch.cat((conv5_out, conv2_out), 1))

        conv7_out = self.conv7(torch.cat((conv6_out, conv1_out), 1))

        up_x = F.interpolate(conv7_out, scale_factor=4, mode="bilinear")


        # tangent line
        x_r = torch.split(up_x, 3, dim=1)

        res =  x_r[0] * x + x_r[1]

        return res