import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

# -------------------------------
# Custom Gaussian Noise Transform
# -------------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, -1.0, 1.0)  # CIFAR100은 [-1, 1] 정규화 후 안전하게 클램핑

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


random.seed(1)
np.random.seed(1)
num_clients = 100
num_classes = 100
dir_path = "Cifar100/"


def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # --------------------------------------------------
    # Transform with Gaussian Noise
    # --------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        AddGaussianNoise(mean=0.0, std=0.05),  # ← 여기에 노이즈 추가
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    for alpha in [0.1, 0.3, 0.5]:
        X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                        niid, balance, partition, 2, alpha)
        train_data, test_data = split_data(X, y)
        save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
                  statistic, niid, balance, partition, 0, alpha)


if __name__ == "__main__":
    generate_cifar100(dir_path, num_clients, num_classes, True, False, "dir")
