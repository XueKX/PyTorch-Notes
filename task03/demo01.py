#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@author: carry
@contact: 864140438@qq.com
@file: demo01.py
@time: 2019/8/11 16:54
@desc: PyTorch基础实现代码
'''
import torch
from torch.autograd import Variable

torch.manual_seed(2)
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.0], [0.0], [1.0], [1.0]]))

# 初始化
w = Variable(torch.Tensor([-1]), requires_grad=True)
b = Variable(torch.Tensor([0]), requires_grad=True)

epochs = 100
costs = []
lr = 0.1

print('before training,predict of x = 1.5 is :')
print('Y_pred = ', float(w.data * 1.5 + b.data > 0))
# w * x_data
# 1 / (1 + torch.exp(-w * x_data))

# 模型训练
for epoch in range(epochs):
    # 计算梯度
    A = 1 / (1 + torch.exp(-(w * x_data + b)))
    # 逻辑损失函数
    J = - torch.mean(y_data * torch.log(A) + (1 - y_data) * torch.log(1 - A))
    # 基础类进行正则化，加上L2范数
    # J = -torch.mean(y_data * torch.log(A) + (1 - y_data) * torch.log(1 - A)) + alpha * w ** 2
    # print(len(J.data))
    # costs.append(J.data.numpy()[0])
    # 自动反向传播
    J.backward()

    # 参数更新
    w.data = w.data - lr * w.grad.data
    w.grad.data.zero_()
    b.data = b.data - lr * b.grad.data
    b.grad.data.zero_()

# 模型测试
print('after trainning,predict of x = 1.5 is :')
print('Y_pred = ', float(w.data * 1.5 + b.data > 0))
print(w.data, b.data)
