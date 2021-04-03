from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def data_load():
    data_train = CIFAR10("./data",download=True,transform=transforms.ToTensor())
    data_test = CIFAR10("./data",download=True,train=False,transform=transforms.ToTensor())
    print(len(data_train), len(data_test))
    train_lodar = DataLoader(data_train, batch_size=72, shuffle=True, num_workers=0)
    test_lodar = DataLoader(data_test, batch_size=1024, num_workers=0)

    return train_lodar,test_lodar
if __name__=="__main__":
    for imgs,tar in data_load()[0]:
        print(imgs.shape)
        print(tar.shape)
        break
    for i in imgs:
        print(i.size())
        # print(i)
        plt.imshow(i.numpy()[0]);
        plt.show()
        break