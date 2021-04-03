# -*- coding = utf-8 -*-
# @time:2020/10/5 18:56
# Author:chen linfeng
# @File:final.py
from collections import OrderedDict
from layersAll import *
import matplotlib.pyplot as plt
from optimizers import *
import numpy as np


class TwoLayerNet:
    def __init__(
            self,
            input_size,
            hidden_size,
            hidden1_size,
            output_size,
            weight_init_std=0.1):
        '''
        初始化权重
        randn生成正太分布的值
        '''
        self.params = {}
        self.params['w1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * \
            np.random.randn(hidden_size, hidden1_size)
        self.params['b2'] = np.zeros(hidden1_size)
        self.params['w3'] = weight_init_std * \
            np.random.randn(hidden1_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['sigmoid'] = Sigmoid()
        # self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
        self.layers['sigmoid1'] = Sigmoid()
        # self.layers['Relu'] = Relu()
        self.layers['Affine3'] = Affine(self.params['w3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict1(self, x):
        '''不使用层'''
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        return y

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss_copy(self, x, t):
        '''原loss'''
        '''
        :param x: 输入数据
        :param t: 监督数据
        :return: 交叉熵损失
        '''
        y = self.predict1(x)
        return cross_entropy_error(y, t)

    def loss(self, x, t):
        '''
        :param x: 输入数据
        :param t: 监督数据
        :return: 交叉熵损失
        '''
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        '''计算正确率函数'''
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(t == y) / float(x.shape[0])
        return accuracy

    def numerical_gradient_tag(self, x, t):
        '''计算权重梯度的函数'''
        def loss_w(w): return self.loss_copy(x, t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        return grads

    def gradinet(self, x, t):
        '''
        计算权重梯度的高速版
        '''
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        # 设置
        grad = {}
        grad['w1'] = self.layers['Affine1'].dw
        grad['b1'] = self.layers['Affine1'].db
        grad['w2'] = self.layers['Affine2'].dw
        grad['b2'] = self.layers['Affine2'].db
        grad['w3'] = self.layers['Affine3'].dw
        grad['b3'] = self.layers['Affine3'].db
        return grad


if __name__ == "__main__":
    # 构建网络
    net = TwoLayerNet(
        input_size=784,
        hidden_size=100,
        hidden1_size=50,
        output_size=10)
    # 获取数据
    x_train, t_train, x_test, t_test = getdata()
    # 超参数设置
    num_epoch = 20000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 优化器设置
    optimizer = SGD(learning_rate)
    optimizer1 = Momentum(lr=learning_rate)
    optimizer2 = Adagrad(learning_rate)
    optimizer3 = Adam(lr=learning_rate)

    # 迭代计算

    for i in range(num_epoch):
        # '''随机获取mini_batch'''
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        # grad = net.numerical_gradient(x_batch,t_batch)
        grad = net.gradinet(x_batch, t_batch)

        # #更新参数
        params = net.params
        # optimizer.update(params,grad)
        optimizer1.update(params, grad)
        # optimizer2.update(params,grad)
        # optimizer3.update(params,grad)

        # 记录学习过程
        loss = net.loss(x_batch, t_batch)
        print(loss)
        train_loss_list.append(loss)
        # 计算每个epoch的识别精度
        if i % batch_size == 0:
            train_acc = net.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            test_acc = net.accuracy(x_test, t_test)
            test_acc_list.append(test_acc)
            print(
                "train_acc, test_acc |" +
                str(train_acc) +
                "," +
                str(test_acc))

    plt.figure(1)
    plt.subplot(221)
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.subplot(222)
    plt.plot(range(len(train_acc_list)), train_acc_list)
    plt.subplot(212)
    plt.plot(range(len(test_acc_list)), test_acc_list)
    plt.show()
