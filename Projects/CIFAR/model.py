import torch.nn as nn
import torch

#CIRAF-10数据集的图像识别
class ResBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride = 1,dow = False):
        super(ResBlock,self).__init__()
        self.dow = dow
        if dow:
            self.one = nn.Conv2d(inchannel,outchannel*2,kernel_size=3,stride=2,padding=1,bias=False)
            self.two = nn.Conv2d(outchannel*2,outchannel*2,kernel_size=3,stride = stride,padding=1,bias=False)
            self.three = nn.Conv2d(inchannel,outchannel*2,kernel_size=1,stride=2,bias=False)
            self.Batch = nn.BatchNorm2d(outchannel*2)
        else:
            self.one = nn.Conv2d(inchannel, outchannel, 3, stride = stride, padding=1, bias=False)
            self.two = nn.Conv2d(inchannel,outchannel,3,stride = stride,padding=1,bias=False)
            self.Batch = nn.BatchNorm2d(outchannel)
        self.conv1 = nn.Sequential(
            self.one,
            self.Batch,
            nn.ReLU(inplace=True),
            self.two,
            self.Batch,
        )
    def forward(self,x):
        out = self.conv1(x)
        if self.dow:
            out += self.Batch(self.three(x))
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.pool =  nn.AvgPool2d(kernel_size=4, stride=1)
        self.linear = nn.Linear(512,10)


        self.layer1 = ResBlock(64,64,stride=1,dow=False)
        self.layer11 = ResBlock(64,64,stride=1,dow=False)

        self.layer2 = ResBlock(64,64,stride=1,dow=True)
        self.layer21 = ResBlock(128, 128, stride=1, dow=False)

        self.layer3 = ResBlock(128,128,stride=1,dow=True)
        self.layer31 = ResBlock(256, 256, stride=1, dow=False)

        self.layer4 = ResBlock(256,256,stride=1,dow=True)
        self.layer41 = ResBlock(512, 512, stride=1, dow=False)

    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer11(x)
        x = self.layer2(x)
        x = self.layer21(x)
        x = self.layer3(x)
        x = self.layer31(x)
        x = self.layer4(x)
        x = self.layer41(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)

        return x

if __name__ =="__main__":
    model = ResNet()
    # print(list(model.named_modules()))
    x = torch.randn(1,3,32,32)
    out = model(x)
    print(out.size())