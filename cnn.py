import torch
from torch import nn
import sys

class five_day_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None


        self.conv_layer1 = nn.Conv2d(in_channels = 1,
                                     out_channels = 64,
                                     kernel_size = (5, 3),
                                     stride = (3, 1),
                                     dilation = (1, 2))
        self.conv_layer2 = nn.Conv2d(in_channels = 64,
                                     out_channels = 128,
                                     kernel_size = (5, 3),
                                     stride = (3, 1),
                                     dilation = (1, 2))

        self.fc = nn.Linear(256, 1)
        self.max_pool = nn.MaxPool2d((2, 1))
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim = -1)


    def forward(self, x):
        x = x.unsqueeze(0)
        x = torch.rand(1, 1, 32, 15).double()
        x = self.leaky_relu(self.conv_layer1(x))
        x = self.max_pool(x)
        x = self.leaky_relu(self.conv_layer2(x))
        x = self.max_pool(x)
        x = torch.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x