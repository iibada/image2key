import torch
import torch.nn as nn
import torch.nn.functional as F
from key_map import base_key


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        pool_size=2,
        pool_stride=2,
    ):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, padding=0),
            nn.Dropout2d(p=0.2),
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = ConvBlock(in_channels=3, out_channels=16)
        self.cnn2 = ConvBlock(in_channels=16, out_channels=32)
        self.cnn3 = ConvBlock(in_channels=32, out_channels=64)
        self.cnn4 = ConvBlock(in_channels=64, out_channels=128)
        self.cnn5 = ConvBlock(in_channels=128, out_channels=256)

        self.fc1 = nn.Linear(in_features=256 * 1 * 6, out_features=512)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer6 = nn.Sequential(self.fc1, nn.ReLU(), nn.Dropout(p=0.2))

        self.fc2 = nn.Linear(
            in_features=512, out_features=len(base_key.all_key_and_type_comb)
        )
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        out = self.cnn5(out)

        out = out.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.fc2(out)

        return out
