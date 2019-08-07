#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@author: carry
@contact: 864140438@qq.com
@file: cifar10_tutorial.py
@time: 2019/8/6 15:41
@desc: CIFAR10 数据集 随机加载 一个batch_size 张展示
@reference: https://pytorch.apachecn.org/docs/1.0/blitz_cifar10_tutorial.html
'''
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


# 定义卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


def download_dataset():
    '''
    加载 CIFAR10 数据集
    :return:
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def imshow(img):
    '''
    输出图像的函数
    :param img:
    :return:
    '''
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def do():
    trainloader, testloader, classes = download_dataset()

    # 随机获取训练图片
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    # 打印图片标签 至少输出5个字符 不够用空格补充
    print(''.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ == '__main__':
    do()
