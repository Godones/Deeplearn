import torch
import torch.nn as nn


class NetLinear(nn.Module):
    def __init__(self,inputfeatures,outputfeatures):
        super(NetLinear,self).__init__()
        self.firstlinear  = nn.Linear(in_features=inputfeatures,out_features=1000)
        self.relu1 = nn.ReLU()
        self.secondlinear = nn.Linear(1000,500)
        self.relu2 = nn.ReLU()
        self.threelinear = nn.Linear(500,outputfeatures)

    def forward(self,x):
        x = self.firstlinear(x)
        x = self.relu1(x)
        x = self.secondlinear(x)
        x = self.relu2(x)
        x = self.threelinear(x)
        return x

if __name__ == '__main__':
    Mylinear= NetLinear(28,10)
    X = torch.randn((28))
    print(X)
    print(Mylinear(X))