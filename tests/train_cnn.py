import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
import pickle

from PIL import Image

import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image

from torchvision import transforms, utils, datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold

input_dir = str(sys.argv[1])  # input dir where candlestick images are stored


# output_dir = str(sys.argv[2])  # output dir to save training model file
# os.makedirs(output_dir, exist_ok=True)


def get_labels(img_dir):
    data = []
    for f in glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True):
        fsplit = f.rsplit("/")
        name = fsplit[-1].rsplit(".")[0]
        fsplit = name.rsplit("_")
        label = fsplit[-1]

        if label == 'UP':
            data.append([f, 0])
        elif label == 'DOWN':
            data.append([f, 1])
        else:
            data.append([f, 2])

        # print(f)

    # column_names = ["images", "trends"]
    df = pd.DataFrame(data)
    return df


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0.0
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
    return train_loss, train_correct


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0.0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct


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


class CandleImageDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.img_labels = get_labels(root)
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # print(img_path)
        # im = Image.open(img_path).convert('RGB')
        image = read_image(img_path)
        # I = np.asarray(im)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image[:3], label


def main():
    print(torchvision.__version__)
    # print(device)

    batch_size = 40
    classes = ('up', 'down', 'side')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(batch_size),
        transforms.ToTensor()
    ])

    # target_transform = Lambda(lambda y: torch.zeros(
    #     3, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    dataset = CandleImageDataset(root=input_dir, transform=transform)

    train_set, test_set = torch.utils.data.random_split(dataset, [1630, 698])

    # Display image and label.
    # train_features, train_labels = next(iter(trainloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    # test_features, test_labels = next(iter(testloader))
    # print(f"Feature batch shape: {test_features.size()}")
    # print(f"Labels batch shape: {test_labels.size()}")

    # img = train_features[0].squeeze()
    # label = train_labels[0]
    #
    # print(f"Label: {classes[label]}")
    # plt.imshow(transforms.ToPILImage()(img))
    # plt.show()
    torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()

    dataset_concat = ConcatDataset([train_set, test_set])

    num_epochs = 100
    k = len(classes)
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_performance = {}

    train_start = time.time()

    for fold, (train_idx, valid_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        fold_start = time.time()

        print('Fold {} start'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(valid_idx)

        trainloader = DataLoader(dataset_concat, batch_size=batch_size, sampler=train_sampler, num_workers=5)
        testloader = DataLoader(dataset_concat, batch_size=batch_size, sampler=test_sampler, num_workers=5)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = Net()
        model.to(device)
        # optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = optim.Rprop(model.parameters(), lr=0.01)  # resilient backpropagation

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(num_epochs):
            train_loss, train_correct = train_epoch(model, device, trainloader, criterion, optimizer)
            test_loss, test_correct = valid_epoch(model, device, testloader, criterion)

            train_loss = train_loss / len(trainloader.sampler)
            train_acc = train_correct / len(trainloader.sampler) * 100
            test_loss = test_loss / len(testloader.sampler)
            test_acc = test_correct / len(testloader.sampler) * 100

            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    test_loss,
                    train_acc,
                    test_acc))

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        fold_end = time.time()
        fold_mins = (fold_end - fold_start) / 60
        print("Fold {} done in {:.2f} minutes.".format(fold + 1, fold_mins))

        fold_performance['fold{}'.format(fold + 1)] = history
    torch.save(model, 'medusa_CNN.pt')
    a_file = open("medusa_history.pkl", "wb")
    pickle.dump(history, a_file)
    a_file.close()

    # a_file = open("data.pkl", "rb")
    # output = pickle.load(a_file)
    train_end = time.time()
    train_mins = (train_end - train_start) / 60
    print("Training done in {:.2f} minutes.".format(train_mins))


if '__main__':
    main()
