import  torchvision
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader

def data_load(batch_size = 64):
    """
    获取训练测试数据
    :return:
    """
    mnist_train = torchvision.datasets.FashionMNIST("./Datasets",train=True,download=True,transform=transforms.ToTensor());
    mnist_test = torchvision.datasets.FashionMNIST("./Datasets",train=False,download=True,transform=transforms.ToTensor());

    # feature ,label = mnist_train[0]
    # print(feature.shape,label)
    # show_image(feature,label)

    train_iter = DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
    test_iter  = DataLoader(mnist_test,batch_size=batch_size)

    return train_iter,test_iter

def get_fashion_mnist_labels(labels):
    """
    获取标签对应的类别
    :param labels:
    :return:
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_image(image,label):
   plt.title(get_fashion_mnist_labels([label])[0])
   plt.imshow(image.view(28,28).numpy())
   plt.show()



if __name__ == '__main__':
    data_load()