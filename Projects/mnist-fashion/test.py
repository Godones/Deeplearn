import torch
from dataload import data_load
from Net import NetLinear
import torch.nn as nn

def test():
    _,test_iter = data_load(64)
    model_pth = "./model.pth"#加载模型
    save_info = torch.load(model_pth)
    model = NetLinear(28*28,10)
    model.load_state_dict(save_info["model"])#读取模型参数
    losser = nn.CrossEntropyLoss()
    model.eval()
    allright = 0
    allx = 0;

    for x,y in test_iter:
        x = x.view(x.shape[0], -1)
        x.to("cuda")
        outy = model(x)
        _, predicted = outy.max(1)
        acc = predicted.eq(y).sum().item() / predicted.shape[0]

        allright +=predicted.eq(y).sum().item()
        allx += predicted.shape[0]
        print("acc: ", acc)

    print("AllAcc: ",allright/allx)
if __name__ == '__main__':
    test()