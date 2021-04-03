from torch.utils import data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize, ToTensor
from torch.utils.data import dataloader
import matplotlib.pyplot as plt

def data_load():
    data_train = MNIST("./data",download=True,transform=
                transforms.Compose(
                    [transforms.Resize((32,32)),
                    transforms.ToTensor()]
                ),train=True)
    data_test = MNIST("./data",download=True,transform=
        transforms.Compose(
            [transforms.Resize((32,32)),
            transforms.ToTensor()]
    ),train=False)
    # print(len(data_train),len(data_test))
    train_lodar = dataloader.DataLoader(data_train,batch_size=256,shuffle=True,num_workers=0)
    test_lodar = dataloader.DataLoader(data_test,batch_size=1024,num_workers=0)
    return train_lodar,test_lodar

if __name__ =="__main__":
    for imgs,tar in data_load()[0]:
        print(imgs.size())
        print(tar.size())
        break
    for i in imgs:
        print(i.size())
        plt.imshow(i.numpy()[0]/255);
        print(i.numpy()[0].shape)
        plt.show()
        break
