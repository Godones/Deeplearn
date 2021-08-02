import torch
from Net import NetLinear
from dataload import data_load
import torch.nn as nn


def train(numepochs):
    myNet  = NetLinear(28*28,10)
    train_iter ,_= data_load(64)
    Lossfunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(myNet.parameters(),lr=0.01)
    for i in range(numepochs):
        for x,y in train_iter:
            x = x.view(x.shape[0],-1)
            x.to("cuda")
            out_y = myNet(x)
            optimizer.zero_grad()
            loss = Lossfunction(out_y,y)
            loss.backward()

            optimizer.step()
            _, predicted = out_y.max(1)
            acc = predicted.eq(y).sum().item()/predicted.shape[0]
            print("numepochs: ",i," loss: ",loss.item()," acc: ",acc)

    save_info = {
        "numepoch":numepochs,
        "optimizer":optimizer.state_dict(),
        "model":myNet.state_dict(),
    }
    torch.save(save_info,"./model.pth")


if __name__ == '__main__':
    train(10)