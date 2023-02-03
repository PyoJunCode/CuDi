import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, scale=1, activation=None):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride=1, bias=bias),
        )
        
        self.scale = scale

        if activation:
            if activation == "ReLU":
                self.activation = nn.ReLU(True)
            elif activation == "GELU":
                self.activation = nn.GELU(True)
            elif activation == "Tanh":
                self.activation = nn.Tanh()
        
    def forward(self, x):
        #S cale
        if self.scale != 1:
            res = F.interpolate(x, scale_factor=self.scale, mode="bilinear")
        else:
            res = x
        res = self.conv_block(res)

        # Activation
        if self.activation:
            res = self.activation(res)
        
        return res

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat         