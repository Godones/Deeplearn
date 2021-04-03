import torch
from model import ResNet
import torch.nn as nn
from data import data_load

_, test_lodar = data_load()#加载数据
model_pth = "./model.pth"#加载模型
save_info = torch.load(model_pth)
model = ResNet()
model.load_state_dict(save_info["model"])#读取模型参数
losser = nn.CrossEntropyLoss()

model.to(device="cuda")
correct = 0
allloss = 0
total = 0
count = 0
# model.eval()
with torch.no_grad():
    for x,target in test_lodar:
        count +=1
        x = x.to(device='cuda')
        target = target.to(device='cuda')
        output = model(x)
        loss = losser(output,target)
        _,predicted = output.max(1)
        all = predicted.eq(target).sum().item()
        correct +=all
        allloss +=loss.item()
        total += target.size(0)
        # print(target[:10],torch.argmax(torch.sigmoid(output),1)[:10])
        print("Loss: ",loss.item()," Acc: ",all/target.size(0))

print("The average loss: {}".format(allloss/count))
print("The average acc: {}".format(correct/total))
