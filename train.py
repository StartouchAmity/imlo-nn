import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralBlock(nn.Module):

    def __init__(self, inputs, outputs, dropout_rate=0.1):
        super().__init__()

        self.convolution1 = nn.Conv2d(inputs, outputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(outputs)
        self.activation1 = nn.ReLU(inplace=True)

        self.convolution2 = nn.Conv2d(outputs, outputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(outputs)
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
        output = self.batchNorm1(output)
        output = self.activation1(output)

        output = self.convolution2(output)
        output = self.batchNorm2(output)

        output += shortcut

        output = self.activation2(output)

        output = self.dropout(output)

        return output

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.neuralBlocks = nn.Sequential(
            NeuralBlock(3, 32, dropout_rate=0.05),
            NeuralBlock(32, 32, dropout_rate=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),

            NeuralBlock(32, 64, dropout_rate=0.1),
            NeuralBlock(64, 64, dropout_rate=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            NeuralBlock(64, 128, dropout_rate=0.15),
            NeuralBlock(128, 128, dropout_rate=0.15),
            nn.MaxPool2d(kernel_size=2, stride=2),

            NeuralBlock(128, 256, dropout_rate=0.2),
            NeuralBlock(256, 256, dropout_rate=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.globalPool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output = self.neuralBlocks(x)

        output = self.globalPool(output)

        output = torch.flatten(output, 1)

        output = self.classifier(output)

        return output