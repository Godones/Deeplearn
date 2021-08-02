import torch.nn as nn
import torch
from torchvision import models

# Yolov1

#Alexnet训练图像分类
class My_Alexnet(nn.Module):
    def __init__(self, features):
        super(My_Alexnet, self).__init__()
        self.feature_select = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),  # 此处应该为局部响应归一化#
            nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),#6*6*256
        )
        self.linears = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6*6*256,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,features)
        )

    def forward(self,x):
        x = self.feature_select(x)
        x = x.view(x.size(0),-1)
        x = self.linears(x)
        return x




class Yolov1_model(nn.Module):
    def __init__(self):
        super(Yolov1_model,self).__init__()
        resnet = models.resnet34(pretrained=True)
        # resnet = models.resnet50(pretrained=True)
        # print(resnet)

        resnet_outfeatures = resnet.fc.in_features #全连接层输入的通道数
        # print(resnet_outfeatures)
        self.twenty_layers = nn.Sequential(*list(resnet.children())[:-2])
        # print(twenty_layers)

        self.conv = nn.Sequential(
            nn.Conv2d(resnet_outfeatures,1024,3,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024,1024,stride=2,padding=1,kernel_size=3,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1,bias= False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.linears = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,7*7*30),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.twenty_layers(x)
        x = self.conv(x)
        x = x.view(x.size()[0],-1)
        x = self.linears(x)
        x = x.reshape(-1,7,7,30)
        return x


if __name__ =="__main__":
    x = torch.randn(1,3,224,224)
    print(x.shape)
    model = My_Alexnet(100)
    print(model(x).shape)
    x = torch.randn(1,3,448,448)
    Yolo = Yolov1_model()
    Yolo(x)
    print(Yolo(x).shape)