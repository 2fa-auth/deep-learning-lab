#!/home/client/Documents/fun/venv/bin/python3
import torch
import torch.nn as nn

import det

class BACKBONEModel(nn.Module):
    def __init__(self, in_chan, out_features=90):
        super().__init__()  
        self.backbone = nn.Sequential(
            nn.Conv2d(in_chan, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(4, 4),
        )
        self.head = nn.Conv2d(128, 90, kernel_size=1)
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x