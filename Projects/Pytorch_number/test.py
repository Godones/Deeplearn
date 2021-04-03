import torch
from model import LeNet
import torch.nn as nn
from data import data_load
import matplotlib.pyplot as plt
import PIL.Image as Im
import PIL
import numpy as np
from matplotlib import image
from torchvision import transforms

_, test_lodar = data_load()#加载数据

def test():
    model_pth = "./model.pth"#加载模型
    save_info = torch.load(model_pth)
    model = LeNet()
    model.load_state_dict(save_info["model"])#读取模型参数
    losser = nn.CrossEntropyLoss()
    correct = 0
    allloss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x,target in test_lodar:
            output = model(x)
            loss = losser(output,target)
            _,predicted = output.max(1)
            all = predicted.eq(target).sum().item()
            correct +=all
            allloss +=loss.item()
            total += target.size(0)
            # print(target[:10],torch.argmax(torch.sigmoid(output),1)[:10])
            print("Loss: ",loss.item()," Acc: ",all/target.size(0))

    print("The average loss: {}".format(allloss))
    print("The average acc: {}".format(correct/total))
    return model

def rand_show():
    for x,target in test_lodar:
        break
    print(x.size(),target.size())
    # for i in x:
    #     print(i.size())
    #     plt.imshow(i.numpy()[0])
    #     plt.show()
    #     break
    print(target[0])
    print(x[0][0][15])
    plt.imshow(x[0].numpy()[0],cmap=plt.cm.gray)
    plt.show()
    return x,target

def self_number():
    pic1 = Im.open("data//6.jpg")
    pic1 = pic1.resize((32,32))
    # pic1 = PIL.ImageOps.invert(pic1)
    plt.imshow(pic1)
    plt.show()
    pic1 = torch.tensor(np.array(pic1),dtype=torch.float32)
    pic1 = torch.transpose(pic1,2,1)
    pic1 = torch.transpose(pic1,1,0)
    pic1.unsqueeze_(0)
    print(pic1.shape)
    return pic1

if __name__ =="__main__":
    model = test()
    x = self_number()
    trans = transforms.Grayscale(num_output_channels=1)
    x = trans(x)
    x = (x-x.min())/(x.max()-x.min())
    print(x)
    output = model(x)
    _, predicted = output.max(1)
    print(predicted)
    plt.imshow(x[0].numpy()[0])

    #
    # x,target =  rand_show()
    # out = model(x[:1])
    # _,pre = out.max(1)
    # print(pre,target[0])