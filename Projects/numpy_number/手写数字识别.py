# -*- coding = utf-8 -*-
# @time:2020/10/3 17:15
# Author:chen linfeng
# @File:Projects.py
import scipy.io as scio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpig
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_error(y, t):
    '''交叉熵误差函数'''
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def softmax(x, batch_size):
    c = np.max(x, axis=1).reshape(batch_size, 1)
    exp_a = np.exp(x - c)  # 防止数据溢出
    sum_exp = np.sum(exp_a, axis=1).reshape(batch_size, 1)
    return exp_a / sum_exp


def getdata():
    data = scio.loadmat('data/mldata/mnist-original.mat')
    x_train, x_test = data['data'][:, :60000] / \
        255, data['data'][:, 60000:] / 255
    t_train, t_test = data['labels'][:, :60000], data['labels'][:, 60000:]

    x_train = x_train.T  # 对数据做转置处理，便于后续操作
    x_test = x_test.T
    t_train = change_onehot(t_train.reshape(60000))
    t_test = change_onehot(t_test.reshape(10000))
    return x_test, t_test


def init_network():
    with open('data/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x, batch_size):  # 读入预置的训练权重
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3, batch_size)
    return y


def change_onehot(x):
    '''将目标值转化为one-hot形式'''
    T = np.zeros((x.shape[0], 10))
    for idx, row in enumerate(T):
        row[int(x[idx])] = 1
        # print(idx, row, x[idx])
    return T


if __name__ == "__main__":
    x, t = getdata()
    print(t.shape, x.shape)
    network = init_network()
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, x.shape[0], batch_size):
        temp_x = x[i:i + batch_size]
        y = predict(network, temp_x, batch_size)
        p = np.argmax(y, axis=1)
        temp_t = t[i:i + batch_size, :]
        temp_t = np.argmax(temp_t, axis=1)
        need = np.sum(temp_t == p)
        accuracy_cnt += np.sum(p == temp_t)
        # print(p.shape,temp_t.shape,i)
        print("loss: " + str(cross_entropy_error(y, t[i:i + batch_size])))

    print(accuracy_cnt)
    print("Accuarcy:" + str(float(accuracy_cnt) / x.shape[0]))
