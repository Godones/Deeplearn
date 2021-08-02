import torch
from model import Yolov1_model
import torch.nn as nn
import torch.optim as op
from Loss import YoloLoss
from data import get_data

torch.cuda.empty_cache()
# print(torch.cuda.is_available())
# 查看模型前半部分是否已经不被训练
# for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(name,':',param.size(),param.requires_grad)


train_loss = 0
correct = 0
total = 0
train_lodar = get_data(32)

model = Yolov1_model()

# model_pth = "./model3.pth"#加载模型继续训练
# save_info = torch.load(model_pth,map_location='cpu')

# model.load_state_dict(save_info["models"])
# model.conv.load_state_dict(save_info["Conv1"])#读取模型参数
# model.linears.load_state_dict(save_info['linars'])

# for i in model.twenty_layers.parameters():
#     i.requires_grad = False

model.train()
model.cuda(device="cuda")
print(torch.cuda.device_count())
losser = YoloLoss(7, 2)
optimizer = op.Adam(model.parameters(), lr=0.000012,weight_decay=1e-9)
# optimizer.load_state_dict(save_info["optimizer"])

print("hi")
torch.cuda.empty_cache()
numepoch = 5
losses = []
# losses = save_info["losses"]
model = nn.DataParallel(model)
for i in range(numepoch):
    for x,target in train_lodar:
        torch.cuda.empty_cache()
        x = x.to(device='cuda')
        target = target.to(device='cuda')
        output = model(x)
        loss = losser(output,target)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        total += target.size(0)
        losses.append(loss.item())
        # correct += predicted.eq(target).sum().item()
        print(i,": ","Loss: ",loss.item())

save_info = {
    "numepoch":numepoch,
    "optimizer":optimizer.state_dict(),
    # "Conv1":model.conv.state_dict(),
    # "linars":model.linears.state_dict(),
    "models":model.state_dict(),
    "losses":losses,
}
torch.save(save_info,"./model3.pth")
