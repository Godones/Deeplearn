import torch
from model import ResNet
import torch.nn as nn
import torch.optim as op
from data import data_load

model = ResNet()
model.train()
losser = nn.CrossEntropyLoss()
optimizer = op.Adam(model.parameters(),lr=0.001,weight_decay=1e-6)
train_loss = 0
correct = 0
total = 0
train_lodar, test_lodar = data_load()

model_pth = "./model.pth"#加载模型继续训练
save_info = torch.load(model_pth)
model = ResNet()
model.load_state_dict(save_info["model"])#读取模型参数
optimizer.load_state_dict(save_info["optimizer"])

numepoch = 100
model.to(device="cuda")
losser.to(device='cuda')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


for i in range(numepoch):
    for x,target in train_lodar:
        x = x.to(device = 'cuda')
        target = target.to(device = 'cuda')

        output = model(x)
        loss = losser(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _,predicted = output.max(1)
        total += target.size(0)
        all = predicted.eq(target).sum().item()
        correct += all
        print(i,"Loss: ", loss.item(), " Acc: ", all / target.size(0))
        # print(i,": ","Loss: ",loss.item())


print(correct/total)
if correct/total > 0.95:
    save_info = {
        "numepoch":numepoch,
        "optimizer":optimizer.state_dict(),
        "model": model.state_dict(),
    }
    torch.save(save_info,"./model.pth")
