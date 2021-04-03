# -*- coding = utf-8 -*-
# @time:2020/10/5 18:56
# Author:chen linfeng
# @File:function.py
import numpy as np
import scipy.io as scio
def function(x):
    '''平方差函数'''
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def _numerical_gradient_no_batch(f, x):
    '''当维度为1的时候'''
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 还原值

    return grad
def numerical_gradient(f, X):
    '''求梯度的函数'''
    if X.ndim == 1:
        '''如果x的维度是1'''
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
    return grad



def sigmoid(x):
    '''sigmoid函数'''
    return 1/(1+np.exp(-x))

def softmax(x):
    '''softmax函数，用于分类问题'''
    c = np.max(x,axis=1).reshape(x.shape[0],1)
    exp_a = np.exp(x-c) ##防止数据溢出
    sum_exp = np.sum(exp_a,axis=1).reshape(x.shape[0],1)
    return exp_a/sum_exp


def cross_entropy_error(y,t):
    '''交叉熵误差函数'''
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


def change_onehot(x):
    '''将目标值转化为one-hot形式'''
    T = np.zeros((x.shape[0], 10))
    for idx, row in enumerate(T):
        row[int(x[idx])] = 1
        # print(idx, row, x[idx])
    return T

def getdata():
    '''
    获取训练集与测试集
    '''
    data = scio.loadmat('data/mldata/mnist-original.mat')
    x_train, x_test = data['data'][:, :60000]/255, data['data'][:, 60000:]/255
    t_train, t_test = data['labels'][:, :60000], data['labels'][: ,60000:]
    print(x_train.shape)

    x_train = x_train.T   ##对数据做转置处理，便于后续操作
    x_test = x_test.T
    t_train = change_onehot(t_train.reshape(60000))
    t_test = change_onehot(t_test.reshape(10000))
    np.random.seed(2020)
    np.random.shuffle(x_train)
    np.random.seed(2020)
    np.random.shuffle(t_train)
    np.random.seed(2020)
    np.random.shuffle(x_test)
    np.random.seed(2020)
    np.random.shuffle(t_test)
    return x_train,t_train,x_test,t_test

if __name__ == "__main__":
    x_train, t_train, x_test, t_test = getdata()