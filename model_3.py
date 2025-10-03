

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

        # Block 2 (Depthwise + 1x1 + normal)
        self.block2 = nn.Sequential(
            nn.Conv2d(16,16,3,padding=1,groups=16,bias=False), # depthwise
            nn.Conv2d(16,24,1,bias=False),                       # pointwise
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24,24,3,padding=1,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24,24,1,bias=False), # 1x1 mix
            nn.Dropout2d(drop_prob)
        )

        # Block 3 (Dilated)
        self.block3 = nn.Sequential(
            nn.Conv2d(24,36,3,padding=2,dilation=2,bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36,36,3,padding=2,dilation=2,bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36,36,3,padding=1,bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36,36,1,bias=False),
            nn.Dropout2d(drop_prob)
        )

        # Block 4 (Stride + dilated)
        self.block4 = nn.Sequential(
            nn.Conv2d(36,48,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48,64,3,padding=2,dilation=2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=2,dilation=2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,1,bias=False),
            nn.Dropout2d(drop_prob)
        )

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64,10)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
