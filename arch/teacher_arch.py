import torch
import torch.nn as nn

from arch.layers import *

class TeacherNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        #self.down = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        
        self.conv1 = nn.Sequential(
            ConvBlock(4, 32, 3, activation="ReLU"),
            ConvBlock(32, 32, 3, activation="ReLU"),
            ConvBlock(32, 32, 3, activation="ReLU"),
        )
        self.conv2 = nn.Sequential(
            ConvBlock(32, 64, 3, activation="ReLU"),
            ConvBlock(64, 64, 3, activation="ReLU"),
            ConvBlock(64, 64, 3, activation="ReLU"),
        )
        self.conv3 = nn.Sequential(
            ConvBlock(64, 128, 3, activation="ReLU"),
            ConvBlock(128, 128, 3, activation="ReLU"),
            ConvBlock(128, 128, 3, activation="ReLU"),
        )
        self.conv4 = nn.Sequential(
            ConvBlock(128, 256, 3, activation="ReLU"),
            ConvBlock(256, 256, 3, activation="ReLU"),
            ConvBlock(256, 256, 3, activation="ReLU"),
        )
        self.conv5 = nn.Sequential(
            ConvBlock(256, 256, 3, activation="ReLU"),
            ConvBlock(256, 256, 3, activation="ReLU"),
            ConvBlock(256, 256, 3, activation="ReLU"),
        )
        self.conv6 = nn.Sequential(
            ConvBlock(384, 128, 3, activation="ReLU", scale=2),
            ConvBlock(128, 128, 3, activation="ReLU"),
            ConvBlock(128, 128, 3, activation="ReLU"),
        )
        self.conv7 = nn.Sequential(
            ConvBlock(192, 64, 3, activation="ReLU", scale=2),
            ConvBlock(64, 64, 3, activation="ReLU"),
            ConvBlock(64, 64, 3, activation="ReLU"),
        )
        self.conv8 = nn.Sequential(
            ConvBlock(96, 32, 3, activation="ReLU", scale=2),
            ConvBlock(32, 32, 3, activation="ReLU"),
            ConvBlock(32, 32, 3, activation="ReLU"),
        )
        self.conv9 = ConvBlock(32, 24, 3, activation="Tanh")

    
    def forward(self, x):

        conv1_out = self.conv1(x)
        conv1_out = F.interpolate(conv1_out, scale_factor=0.5, mode="bilinear")
        conv2_out = self.conv2(conv1_out)
        conv2_out = F.interpolate(conv2_out, scale_factor=0.5, mode="bilinear")
        conv3_out = self.conv3(conv2_out)
        conv3_out = F.interpolate(conv3_out, scale_factor=0.5, mode="bilinear")

        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)

        conv6_out = self.conv6(torch.cat((conv3_out, conv5_out), dim=1))
        conv7_out = self.conv7(torch.cat((conv2_out, conv6_out), dim=1))
        conv8_out = self.conv8(torch.cat((conv1_out, conv7_out), dim=1))

        conv9_out = self.conv9(conv8_out)


        # High order curve estimation
        x_r = torch.split(conv9_out, 3, dim=1)

        res = x[:, :3, :, :]
        for i in range(8):
            res = res + x_r[i] * (torch.pow(res, 2) - res)

        r = torch.cat(x_r, 1)

        return res, r