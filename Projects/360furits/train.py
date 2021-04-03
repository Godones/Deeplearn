import torch
from model import My_Alexnet
import torch.nn as nn
import torch.optim as op
from image_load import data_load

model = My_Alexnet(131)
model.train()
losser = nn.CrossEntropyLoss()
optimizer = op.Adam(model.parameters(),lr=0.002,weight_decay=1e-6)
train_loss = 0
correct = 0
total = 0
train_lodar, test_lodar = data_load()

model_pth = "./model.pth"#加载模型继续训练
save_info = torch.load(model_pth)
model = My_Alexnet(131)
model.load_state_dict(save_info["model"])#读取模型参数
optimizer.load_state_dict(save_info["optimizer"])

numepoch = 1
# model.to(device="cuda")
# losser.to(device='cuda')

for i in range(numepoch):
    for x,target in train_lodar:
        # x = x.to(device = 'cuda')
        # target = target.to(device = 'cuda')
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

save_info = {
    "numepoch":numepoch,
    "optimizer":optimizer.state_dict(),
    "model": model.state_dict(),
}
torch.save(save_info,"./model.pth")
