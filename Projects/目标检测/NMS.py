import numpy as np
import matplotlib.pyplot as plt
import torch
from Loss import YoloLoss, calculate_iou
from data import name
import  cv2
from data import get_data
from  model import Yolov1_model
import PIL.Image as Image

def transBox(labels):
    """
    将 labels转换为98*25,98个框，25为4个坐标，一个置信度，20个分类类别
    :param labels: 7*7*30
    :return: NMS(bbox)
    """
    grids = 7
    bbox = torch.zeros((98, 25))
    for i in range(7):
        for j in range(7):
            bbox[2 * (i * 7 + j), 0:4] = torch.tensor([(labels[i, j, 0] + j) / grids - labels[i, j, 2] / 2,
                                                       (labels[i, j, 1] + i) / grids - labels[i, j, 3] / 2,
                                                       (labels[i, j, 0] + j) / grids + labels[i, j, 2] / 2,
                                                       (labels[i, j, 1] + i) / grids + labels[i, j, 3] / 2])
            bbox[2 * (i * 7 + j), 5:] = labels[i, j, 10:]
            bbox[2 * (i * 7 + j), 4] = labels[i, j, 4]
            bbox[2 * (i * 7 + j) + 1, 0:4] = torch.tensor([(labels[i, j, 5] + j) / grids - labels[i, j, 7] / 2,
                                                           (labels[i, j, 6] + i) / grids - labels[i, j, 8] / 2,
                                                           (labels[i, j, 5] + j) / grids + labels[i, j, 7] / 2,
                                                           (labels[i, j, 6] + i) / grids + labels[i, j, 8] / 2])
            bbox[2 * (i * 7 + j) + 1, 5:] = labels[i, j, 10:]
            bbox[2 * (7 * i + j) + 1, 4] = labels[i, j, 9]

    # print(bbox.size())
    return NMS(bbox)


def NMS(bbox: torch.Tensor, conf_classify=0.3, conf_iou=0.4):
    """

    :param bbox:
    :param conf_classify:
    :param conf_iou:
    :return:
    """
    n = bbox.size()[0]
    bbox_classify = bbox[:, 5:]
    bbox_confi = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_classify)
    bbox_classify_confi = bbox_confi * bbox_classify  # 综合置信度 = 分类概率*置信度
    bbox_classify_confi[bbox_classify_confi < conf_classify] = 0 #去掉哪些小于分类阈值的

    for c in range(20):#对20个类别分别进行筛选
        #从大到小排序
        rank = torch.sort(bbox_classify_confi[:, c], descending=True).indices
        for i in range(98):
            if bbox_classify_confi[rank[i], c] != 0:
                for j in range(i + 1, 98):
                    if bbox_classify_confi[rank[j], c] != 0:
                        iou = calculate_iou(bbox[rank[i], 0:4], bbox[rank[j], 0:4])
                        if iou > conf_iou:
                            bbox_classify_confi[rank[j], c] = 0
    #将类别中综合置信度为0的框去掉
    bbox = bbox[torch.max(bbox_classify_confi,dim=1).values > 0]
    bbox_classify_confi = bbox_classify_confi[torch.max(bbox_classify_confi,dim=1).values>0]

    final_box = torch.ones((bbox.size()[0]),6)
    final_box[:,1:5] = bbox[:,0:4]#坐标信息
    final_box[:,0] = torch.argmax(bbox[:,5:],dim=1).int()#对应的类别
    final_box[:,5] = torch.max(bbox_classify_confi,dim=1).values #综合置信度

    return final_box

def draw(img, bbox):
    """
    画出检测图像
    :param img:
    :param bbox:
    :return:
    """
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    print(img)
    pic_name = name()
    h,w = img.shape[0:2]
    n = bbox.size()[0]
    for i in range(n):
        p1 = (int(w*bbox[i,1]),int(h*bbox[i,2]))
        p2 = (int(w*bbox[i,3]),int(h*bbox[i,4]))
        obj_name = pic_name[int(bbox[i,0])]
        confidence = bbox[i,5]
        cv2.rectangle(img,p1,p2,(0,255,255,2))
        cv2.putText(img,obj_name,p1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv2.imshow("bbox",img)
        cv2.waitKey(0)

def test():
    model = Yolov1_model()
    model_pth = "./model2.pth"  # 加载模型继续训练
    save_info = torch.load(model_pth, map_location='cpu')
    model.conv.load_state_dict(save_info["Conv1"])  # 读取模型参数
    model.linears.load_state_dict(save_info['linars'])
    model.eval()
    # model.to("cuda")
    train_loarder = get_data(1)

    for i, (inputs, labels) in enumerate(train_loarder):
        # inputs = inputs.cuda()
        # 以下代码是测试labels2bbox函数的时候再用
        labels = labels.float().cuda()
        labels = labels.squeeze(dim=0)
        pred = model(inputs)  # pred的尺寸是(1,7,7,30)
        pred = pred.squeeze(dim=0)  # 压缩为(7,7,30)

        ## 测试labels2bbox时，使用 labels作为labels2bbox2函数的输入
        bbox = transBox(pred)
        # 此处可以用labels代替pred，测试一下输出的bbox是否和标签一样，从而检查labels2bbox函数是否正确。
        # 当然，还要注意将数据集改成训练集而不是测试集，因为测试集没有labels。
        inputs = inputs.squeeze(dim=0)  # 输入图像的尺寸是(1,3,448,448),压缩为(3,448,448)
        inputs = inputs.permute((1, 2, 0))  # 转换为(448,448,3)
        img = inputs.cpu().numpy()
        img = (255 * img)%255  # 将图像的数值从(0,1)映射到(0,255)并转为非负整形
        img = img.astype(np.uint8)
        draw(img, bbox.cpu())  # 将网络预测结果进行可视化，将bbox画在原图中，可以很直观的观察结果
        print(bbox.size(), bbox)

if __name__ == "__main__":
    x = torch.randn(7,30,30)
    y = transBox(x)
    print(y.shape)
    test()
