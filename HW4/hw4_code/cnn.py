import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the convolutional layers and activation functions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.2)

        # Define the fully connected layers and activation functions
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=256)
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.dropout4 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Pass input through convolutional layers
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # Pass through fully connected layers
        x = self.avgpool(x)
        x = self.flatten(x)

        x = F.leaky_relu(self.fc1(x))
        x = self.dropout3(x)

        x = F.leaky_relu(self.fc2(x))
        x = self.dropout4(x)

        # Final output layer with softmax activation
        output = F.softmax(self.fc3(x), dim=1)
        return output
