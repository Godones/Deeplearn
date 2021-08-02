import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
from data import get_data


def calculate_iou(bbox1, bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0.0, 0.0, 0.0, 0.0]  # bbox1和bbox2的交集
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0], bbox2[0])
        intersect_bbox[1] = max(bbox1[1], bbox2[1])
        intersect_bbox[2] = min(bbox1[2], bbox2[2])
        intersect_bbox[3] = min(bbox1[3], bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # print(area_intersect)
    if area_intersect > 0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0


class YoloLoss(nn.Module):
    def __init__(self, grids, bboxnum):
        super(YoloLoss, self).__init__()
        self.grids = grids  # 7代表将图像分为7x7的网格
        self.bboxnum = bboxnum  # 2代表一个网格预测两个框
        self.l_coord = 5  # 5代表 λcoord  更重视8维的坐标预测
        self.l_noobj = 0.5  # 0.5代表没有object的bbox的confidence loss

    def forward(self, pred_tensor, target_tensor):
        #   pred_tensor :size(batchsize, S, S, Bx5 + 20 = 30) [x, y, w, h, c]
        #   target_tensor: (tensor) size(batchsize,S,S,30)
        # coo_mask = target_tensor[:,:,:,4]>0 #找到含有目标的索引
        # coo_mask = torch.unsqueeze(coo_mask,-1).expand(target_tensor.size())
        #
        # non_mask = target_tensor[:,:,:,4]==0 #没有目标标签的索引
        # non_mask = torch.unsqueeze(non_mask,-1).expand(target_tensor.size())
        xy_loss = 0  # xy坐标的损失（含有目标的）
        wh_loss = 0  # w,h的损失

        allloss = 0
        coo_loss = 0  # 含有目标的损失，包含置信度损失，类别损失，
        nonloss = 0  # 不含目标的损失，只有置信度损失
        classloss = 0
        pred_tensor = pred_tensor.astype(torch.float)
        target_tensor = target_tensor.astype(torch.float)
        for i in range(pred_tensor.size()[0]):
            for j in range(7):
                for k in range(7):
                    if target_tensor[i, j, k, 4] > 0:
                        # coo_target = target_tensor[i][coo_mask[i]].view(-1,30)
                        # box_target = coo_target[:,0:5].contiguous().view(-1,5)#含有目标的坐标与w,h
                        # box_pred = pred_tensor[i,:,:,:5].view(-1,5)
                        # box_pred2 = pred_tensor[i,:,:,5:10].view(-1,5)
                        # 坐标转化
                        pred_box1 = ((pred_tensor[i, j, k, 0] + j) / self.grids - pred_tensor[i, j, k, 2] / 2,
                                     (pred_tensor[i, j, k, 1] + k) / self.grids - pred_tensor[i, j, k, 3] / 2,
                                     (pred_tensor[i, j, k, 0] + j) / self.grids + pred_tensor[i, j, k, 2] / 2,
                                     (pred_tensor[i, j, k, 1] + k) / self.grids + pred_tensor[i, j, k, 3] / 2)

                        pred_box2 = ((pred_tensor[i, j, k, 5] + j) / self.grids - pred_tensor[i, j, k, 7] / 2,
                                     (pred_tensor[i, j, k, 6] + k) / self.grids - pred_tensor[i, j, k, 8] / 2,
                                     (pred_tensor[i, j, k, 5] + j) / self.grids + pred_tensor[i, j, k, 7] / 2,
                                     (pred_tensor[i, j, k, 6] + k) / self.grids + pred_tensor[i, j, k, 8] / 2)

                        target_box = ((target_tensor[i, j, k, 0] + j) / self.grids - target_tensor[i, j, k, 2] / 2,
                                      (target_tensor[i, j, k, 1] + k) / self.grids - target_tensor[i, j, k, 3] / 2,
                                      (target_tensor[i, j, k, 0] + j) / self.grids + target_tensor[i, j, k, 2] / 2,
                                      (target_tensor[i, j, k, 1] + k) / self.grids + target_tensor[i, j, k, 3] / 2)

                        # print(coo_target.size(),box_pred.size())
                        # print(box_target)
                        # print(j,k)
                        # print(target_box)
                        # print(pred_box1)
                        # print(pred_box2)
                        iou1 = calculate_iou(pred_box1, target_box)
                        iou2 = calculate_iou(pred_box2, target_box)
                        # print(iou1,iou2)
                        if iou1 >= iou2:
                            coo_loss += self.l_coord * (
                                        torch.sum((pred_tensor[i, j, k, :2] - target_tensor[i, j, k, :2]) ** 2) +
                                        torch.sum((pred_tensor[i, j, k, 2:4].sqrt() - target_tensor[i, j, k,
                                                                                      2:4].sqrt()) ** 2))
                            coo_loss += 1 * torch.sum((pred_tensor[i, j, k, 4] - iou1) ** 2)
                            nonloss += self.l_noobj * (pred_tensor[i, j, k, 4] - iou2) ** 2
                        else:
                            coo_loss += self.l_coord * (
                                    torch.sum((pred_tensor[i, j, k, 5:7] - target_tensor[i, j, k, 5:7]) ** 2) +
                                    torch.sum((pred_tensor[i, j, k, 7:9].sqrt() - target_tensor[i, j, k,
                                                                                  7:9].sqrt()) ** 2))
                            coo_loss += torch.sum(1 * (pred_tensor[i, j, k, 9] - iou2) ** 2)
                            nonloss += self.l_noobj * (pred_tensor[i, j, k, 9] - iou1) ** 2
                        classloss += torch.sum((pred_tensor[i, j, k, 10:] - target_tensor[i, j, k, 10:]) ** 2)

                    else:
                        nonloss += torch.sum(self.l_noobj * (pred_tensor[i, j, k, [4, 9]] ** 2))

        allloss = nonloss + coo_loss + classloss
        return allloss / pred_tensor.size()[0]


if __name__ == "__main__":
    Loss = YoloLoss(7, 2)
    t = torch.tensor([[1, 2, -1, 2], [2, 3, 4, 5]])
    # Loss.calculate_iou(t,t2)
    # t = torch.unsqueeze(t[:,:2],1).expand(2,2,2)
    # t2 = torch.unsqueeze(t2[:,:2],1).expand(2,2,2)
    print(t)
    print(t[:, 2] > 0)
    coo = t[:, 2] > 0
    coo = torch.unsqueeze(coo, -1).expand(t.size())
    print(coo)
    print(t[coo])

    # test
    dataloader = get_data()
    for img, target in dataloader:
        print("!", img.shape)
        print("!", target.shape)
        Loss(torch.rand(32, 7, 7, 30), target)
        # print(target[target[:,:,:,4]>0])
        break
