import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralBlock(nn.Module):

    def __init__(self, inputs, outputs, dropout_rate=0.5):
        super().__init__()

        self.convolution1 = nn.Conv2d(inputs, outputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(outputs)
        self.activation1 = nn.ReLU(inplace=True)

        self.convolution2 = nn.Conv2d(outputs, outputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(outputs)
        if inputs != outputs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outputs)
            )
        else:
            self.shortcut = nn.Identity()
        self.activation2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):

        shortcut = self.shortcut(x)

        output = self.convolution1(x)
        output = self.batchnorm1(output)
        output = self.activation1(output)

        output = self.convolution2(output)
        output = self.batchnorm2(output)

        output += shortcut

        output = self.activation2(output)

        output = self.dropout(output)

        return output