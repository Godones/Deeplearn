import torch
import torch.nn as nn
import numpy as np
import torch.optim.functional as f
class YoloLoss(nn.Module):
    def __init__(self,grids,bboxnum):
        super(YoloLoss,self).__init__()
        self.grids = grids  # 7代表将图像分为7x7的网格
        self.bboxnum = bboxnum  # 2代表一个网格预测两个框
        self.l_coord = 5 # 5代表 λcoord  更重视8维的坐标预测
        self.l_noobj = 0.5  # 0.5代表没有object的bbox的confidence loss

    def calculate_iou(self,bbox1, bbox2):
        """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
        intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
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
    def forward(self,pred_tensor,target_tensor):
        #   pred :size(batchsize, S, S, Bx5 + 20 = 30) [x, y, w, h, c]
        #   target_tensor: (tensor) size(batchsize,S,S,30)
        pass


if __name__=="__main__":
    Loss = YoloLoss(7,2)
    t = torch.tensor([[1,1,2,2],[2,3,4,5]])
    t2 = torch.tensor([1,1,4,4])
    # Loss.calculate_iou(t,t2)
    # t = torch.unsqueeze(t[:,:2],1).expand(2,2,2)
    # t2 = torch.unsqueeze(t2[:,:2],1).expand(2,2,2)
    # print(t)
    # print(t2)
    print(t[:,2]>0)
    print(t[t[:,2]>0])