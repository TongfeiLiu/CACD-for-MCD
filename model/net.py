import torch
import torch.nn as nn

import torch
from torch.nn import Module, Conv2d, Parameter, Softmax


class CAE(nn.Module):
    def __init__(self, in_channels, patch_size):
        super().__init__()
        self.en_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.en_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        hw = patch_size **2

        self.feature = nn.Sequential(
            nn.Linear(64*hw, 20),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear(20, 64*hw),
            nn.ReLU(inplace=True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, patch_size, patch_size))

        self.de_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.de_conv1 = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.en_conv1(x)
        x = self.en_conv2(x)
        x = self.flatten(x)
        feature = self.feature(x)
        x = self.linear(feature)
        x = self.unflatten(x)
        x = self.de_conv2(x)
        x = self.de_conv1(x)
        return feature, x

class COAE(nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, in_channels),
        )

    def forward(self, x):
        x_hat = self.fc(x)
        return x_hat