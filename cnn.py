import torch
from torch import nn
import sys

class five_day_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer1 = nn.Conv2d(in_channels = 1,
                                     out_channels = 64,
                                     kernel_size = (5, 3),
                                     stride = (3, 1),
                                     dilation = (2, 1),
                                     padding = (12, 1))

        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv_layer2 = nn.Conv2d(in_channels = 64,
                                     out_channels = 128,
                                     kernel_size = (5, 3),
                                     stride = (3, 1),
                                     dilation = (2, 1),
                                     padding=(12, 1))

        self.batch_norm2 = nn.BatchNorm2d(128)


        self.conv_layer3 = nn.Conv2d(in_channels = 128,
                                     out_channels = 256,
                                     kernel_size = (5, 3),
                                     stride = (3, 1),
                                     dilation = (2, 1),
                                     padding = (12, 1))

        self.batch_norm3 = nn.BatchNorm2d(256)


        self.fc = nn.Linear(46080, 2)
        self.dropout = nn.Dropout(0.5)
        self.max_pool = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.leaky_relu(self.batch_norm1(self.conv_layer1(x)))
        x = self.max_pool(x)
        x = self.leaky_relu(self.batch_norm2(self.conv_layer2(x)))
        x = self.max_pool(x)
        x = self.leaky_relu(self.batch_norm3(self.conv_layer3(x)))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1, -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x