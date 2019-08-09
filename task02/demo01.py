#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@author: carry
@contact: 864140438@qq.com
@file: demo01.py
@time: 2019/8/8 14:29
@desc:
'''
import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 5  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1
DOWNLOAD_MNIST = False
LR = 0.001

# Mnist digits dataset
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class Logits(nn.Module):
    def __init__(self):
        super(Logits, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)
        self.sigmoid = nn.Sigmoid()+6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x


test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
    torch.FloatTensor).cuda() / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels

alpha = 0.001

logits = Logits().cuda()
# optimizer = torch.optim.SGD(logits.parameters(), lr=LR)  # optimize all cnn parameters
# optimizer.zero_grad()
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

Accurate = []
Astore = []
bstore = []
A, b = [i for i in logits.parameters()]
A.cuda()
b.cuda()
for e in range(EPOCH):
    for step, (x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = x.view(-1, 28 * 28).cuda()  # reshape x to (batch, time_step, input_size)
        b_y = b_y.cuda()

        output = logits(b_x)  # logits output
        loss = loss_func(output, b_y)  # cross entropy loss
        if A.grad is not None:
            A.grad.zero_()
            b.grad.zero_()
        loss.backward()  # backpropagation, compute gradients

        A.data = A.data - alpha * A.grad.data
        b.data = b.data - alpha * b.grad.data
        if step % 1500 == 0:
            test_output = logits(test_x.view(-1, 28 * 28))
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            Accurate.append(sum(test_y.cpu().numpy() == pred_y.cpu().numpy()) / (1.0 * len(test_y.cpu().numpy())))
            print(Accurate[-1])
            Astore.append(A.detach())
            bstore.append(b.detach())
test_output = logits(test_x.view(-1, 28 * 28))
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()

print(pred_y, 'prediction number')
print(test_y, 'real number')
Accurate.append(sum(test_y.cpu().numpy() == pred_y.cpu().numpy()) / (1.0 * len(test_y.cpu().numpy())))
print(Accurate[-1])

for i in range(len(Astore)):
    Astore[i] = (Astore[i] - Astore[-1]).norm()
    bstore[i] = (bstore[i] - bstore[-1]).norm()

plt.plot(Astore, label='A')
plt.plot(bstore, label='b')
plt.legend()
plt.show()
plt.cla()
plt.plot(Accurate)

plt.show()
