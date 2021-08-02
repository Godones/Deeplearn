import torch.nn as nn
import torch

#先用Alexnet训练图像分类
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
#
if __name__ =="__main__":
    x = torch.randn(1,3,224,224)
    print(x.shape)
    model = My_Alexnet(100)
    print(model(x))