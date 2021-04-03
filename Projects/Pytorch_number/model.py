import torch.nn as nn
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(16*6*6,120)
        self.linear2 = nn.Linear(120,84)
        self.linear3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

if __name__ =="__main__":
    model = LeNet()
    ret = model(torch.randn(1,3,32,32))
    losser = nn.CrossEntropyLoss()
    target = torch.randn(1).long()
    print(ret,target)
    print(losser(ret,target))
    print(torch.sigmoid(ret))