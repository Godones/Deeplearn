from torchvision.datasets import ImageFolder
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import  numpy as np
def data_load():
    data_path = "../360furits/data/fruits-360/Test"
    data_path2 = "../360furits/data/fruits-360/Training"
    data_train = ImageFolder(data_path2,transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224))]
    ))
    data_test = ImageFolder(data_path,transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224))]
    ))

    train_lodar = DataLoader(data_train, batch_size=32, shuffle=True)
    test_lodar = DataLoader(data_test, batch_size=100)
    return train_lodar,test_lodar
if __name__=="__main__":
    for imgs,tar in data_load()[0]:
        print(imgs.shape)
        print(tar.shape)
        break
    for i in imgs:
        print(i.size())
        print(tar)
        print(i)
        plt.imshow(i.numpy().transpose(1,2,0).astype(np.float32))
        plt.show()
        break