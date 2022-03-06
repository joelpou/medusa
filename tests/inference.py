import os
import sys
import torch
import pandas as pd
from PIL import Image
import numpy as np
import glob
import time
import pickle
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.io import read_image
from torchvision import transforms
import torchvision.transforms as T
import matplotlib.pyplot as plt


# input_dir = str(sys.argv[1])  # input dir where candlestick images are stored


class Net(nn.Module):
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
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def main():
    print(torchvision.__version__)
    batch_size = 40

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(batch_size),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.to(device)
    # image = read_image("../images/DOWN/A_2013-02-14_2013-02-27_9_DOWN.png")
    image = read_image("../images/SIDE/A_2013-03-21_2013-04-04_10_SIDE.png")
    # image = read_image("../images/UP/A_2013-01-02_2013-01-09_6_UP.png")

    img = image.to(device)

    transformed = transform(img[:3]).to(device)
    # timg = T.ToPILImage()(transformed.to('cpu'))
    # plt.imshow(np.asarray(timg))
    # plt.show()

    # # model.load_state_dict(torch.load("./medusa_CNN0.pt"))
    model = torch.load("./medusa_CNN0.pt")
    model.eval()
    #
    with torch.no_grad():
        output = model.forward(transformed[None, ...])  # insert singleton batch since model expects batch input
    scores, prediction = torch.max(output.data, 1)

    # output = model(transformed[None, ...])
    print(prediction)

    # model = Net()
    # model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # optimizer = optim.Rprop(model.parameters(), lr=0.01)  # resilient backpropagation


if '__main__':
    main()
