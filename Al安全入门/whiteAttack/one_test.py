
import torch
import torchvision
from torchvision import datasets,transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def load():
    data_train = datasets.MNIST(root="./data/",
                                transform=transforms.ToTensor(),
                                train=True,
                                download=True)
    data_test = datasets.MNIST(root="./data/",
                               transform=transforms.ToTensor(),
                               train=False,
                               download=True)
    batch_size = 64
    train_loder = torch.utils.data.DataLoader(
        dataset = data_train,
        batch_size = batch_size,
        shuffle = True
    )
    test_loder = torch.utils.data.DataLoader(
        dataset = data_test,
        batch_size = batch_size,
        shuffle = True
    )
    return train_loder,test_loder

def show(train_loder):
    pic = None
    label = None
    for picture in train_loder:
        pic,label = picture
        print(label,pic.size())
        print(pic[0].size())
        print(pic.size())

        img = torchvision.utils.make_grid(pic)
        img = img.numpy().transpose(1, 2, 0)
        std = [0.5, 0.5, 0.5]
        mean = [0.5, 0.5, 0.5]
        img = img * std + mean
        plt.imshow(img)
        plt.show()
        print(img.shape)
        break
    return pic,label


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2)
        )
        self.dense = nn.Sequential(
            nn.Linear(14*14*128,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,10)
        )
        self.linears = nn.Sequential(
            nn.Linear(784,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(50,10)
        )
    def forward(self,x):
        # x = self.conv1(x)
        # x = x.view(-1,14*14*128)
        # x = self.dense(x)
        x = self.linears(x)
        return x

if __name__ == "__main__":
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    train_loder,test_loder = load()
    pic,label = show(train_loder)
    img1 = pic[0].numpy()
    img1.shape
    img1 = img1.transpose(1,2,0)*std+mean
    plt.imshow(img1)


    model = Model()
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())


    print(model)

    n_epoch = 5
    for epoch in range(n_epoch):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch,n_epoch))
        loss = 0.0
        for data in train_loder:
            x_train,y_train = data
            outputs = model(x_train)
            _,pred = torch.max(outputs.data,1)
            optimizer.zero_grad()

            # print(outputs)

            loss = cost(outputs,y_train)

            loss.backward()
            optimizer.step()
        test_correct = 0
        for data in test_loder:
            x_test,y_test = data
            outputs = model(x_test)
            _,pred = torch.max(x_test)
            test_correct = torch.sum(pred == y_test.data)
        print("Loss :{}, Testcorresc: {}".format(loss,test_correct))
