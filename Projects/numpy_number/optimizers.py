# -*- coding = utf-8 -*-
# @time:2020/10/9 21:00
# Author:chen linfeng
# @File:optimizers.py


import numpy as np
class SGD:
    def __init__(self, lr=0.13):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    '''
    Momentum可以加快收敛速度
    '''
    def __init__(self, lr=0.01, alpha=0.9):
        self.lr = lr
        self.v = None
        self.alpha = alpha

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)
        for key in params.keys():
            self.v[key] = self.alpha * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class Adagrad:
    '''
    Adagrad 方法记录以前梯度的平方和，在更新参数时，通过乘以一个系数，可以调整学习的尺度
    当参数的导数很大时，可以降低其学习率，使其波动幅度减小
    '''

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    '''此方法结合了momentum 与 Adagrad的优点'''

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 **
                                 self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
