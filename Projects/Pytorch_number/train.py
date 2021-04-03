import torch
from model import LeNet
import torch.nn as nn
import torch.optim as op
from data import data_load

model = LeNet()
model.train()
losser = nn.CrossEntropyLoss()
optimizer = op.Adam(model.parameters(),lr=0.01)
train_loss = 0
correct = 0
total = 0
train_lodar, test_lodar = data_load()

numepoch = 3
for i in range(numepoch):
    for x,target in train_lodar:
        output = model(x)
        loss = losser(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _,predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        print(i,": ","Loss: ",loss.item())


save_info = {
    "numepoch":numepoch,
    "optimizer":optimizer.state_dict(),
    "model": model.state_dict(),
}
torch.save(save_info,"./model.pth")
