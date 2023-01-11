import torch
import torch.nn as nn
import torch.nn.functional as F

class CovNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (2, 2))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, (2, 2))
        self.pool2 = nn.MaxPool2d(1, 2)
        self.conv3 = nn.Conv2d(12, 3, (1, 1))

        self.fc1 = nn.Linear(243, 120)
        self.fc2 = nn.Linear(120, 3)

    def forward(self, x):
        # save_image(x, 'first0.png')
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # The output from the convolutional layers represents high-level features in the data. 
        # While that output could be flattened and connected to the output layer, adding a 
        # fully-connected layer is a (usually) cheap way of learning non-linear combinations 
        # of these features.
        x = self.fc1(x)
        x = self.fc2(x)
        return x