from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import torch

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
           'train', 'bottle', 'chair', 'dining table', 'potted plant',
           'sofa', 'tvmonitor']

def name():
    CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
               'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
               'train', 'bottle', 'chair', 'dining table', 'potted plant',
               'sofa', 'tvmonitor']
    return CLASSES
# voc_trainset = VOCDetection(
#     ".\data\\train", year="2007", image_set='trainval', download=False)
# print(len(voc_trainset))
# voc_testset = VOCDetection(".\data\\test", year="2007",
#                            image_set='test', download=False)
# print(len(voc_testset))

Dataset_path = "./data/train/VOCdevkit/VOC2007/"
Dataset_path2 = "./data/test/VOCdevkit/VOC2007/"

def show_rect(image: np.ndarray, bndbox):
    pt1 = bndbox[:2]
    pt2 = bndbox[2:]
    image_show = image
    return cv2.rectangle(image_show, pt1, pt2, (0, 255, 255), 2)

def show_name(image, name, p_tl):
    return cv2.putText(image, name, p_tl, 1, 1, (255, 255, 0))

def show(voc_trainset):
    for i, samele in enumerate(voc_trainset, 1):
        image, annotation = samele[0], samele[1]["annotation"]
        objects = annotation['object']
        show_image = np.array(image)
        print("{} objects_num: {}".format(i, len(objects)))
        if len(objects) == 1:
            objects = objects[0]
            object_name = objects['name']
            object_bndbox = objects['bndbox']
            x_min = int(object_bndbox['xmin'])
            y_min = int(object_bndbox['ymin'])
            x_max = int(object_bndbox['xmax'])
            y_max = int(object_bndbox['ymax'])
            show_image = show_rect(show_image, (x_min, y_min, x_max, y_max))
            show_image = show_name(show_image, object_name, (x_min, y_min))
        else:
            for j in objects:
                object_name = j['name']
                object_bndbox = j['bndbox']
                x_min = int(object_bndbox['xmin'])
                y_min = int(object_bndbox['ymin'])
                x_max = int(object_bndbox['xmax'])
                y_max = int(object_bndbox['ymax'])
                show_image = show_rect(show_image, (x_min, y_min, x_max, y_max))
                show_image = show_name(show_image, object_name, (x_min, y_min))
        cv2.imshow('image', show_image)
        cv2.waitKey(0)


class Voc2007(Dataset):
    """重写Dataset"""
    def __init__(self,filename,train = "train",transforms = True):
        Dataset_path = filename
        self.filenames = []
        if train=="train":
            with open(Dataset_path + "ImageSets/Main/train.txt", 'r') as f:  # 调用包含训练集图像名称的txt文件
                self.filenames = [x.strip() for x in f]
        elif train =="trainval":
            with open(Dataset_path + "ImageSets/Main/trainval.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        else:
            with open(Dataset_path + "ImageSets/Main/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]

        self.img_path = Dataset_path + "JPEGImages/"  # 原始图像所在的路径
        self.label_path = "./labels/"  # 图像对应的label文件(.txt文件)的路径
        self.transforms = transforms
        # print(len(self.filenames),self.img_path,self.label_path)
    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, indx):
        img = cv2.imread(self.img_path + self.filenames[indx] + ".jpg")  # 读取原始图像
        h, w = img.shape[0:2]
        input_size = 448
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        # 输入YOLOv1网络的图像尺寸为448x448
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
        if h > w:
            padw = (h - w) // 2
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
        elif w > h:
            padh = (w - h) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)

        img = cv2.resize(img, (input_size, input_size))
        # 图像增广部分，将其转换为torch.tensor
        if self.transforms:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        img = transform(img)
        # print(img.size())
        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        with open(self.label_path + self.filenames[indx] + ".txt") as f:
            bbox = f.read().split('\n')

        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox) % 5 != 0:
            raise ValueError("File:" + self.label_path + self.filenames[indx] + ".txt" + "——bbox Extraction Error!")

        # print(len(bbox))
        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据
        #bbox[0]是分类，[1-4]是坐标
        for i in range(len(bbox) // 5):
            if padw != 0:
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
            elif padh != 0:
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w

        # 验证一下，查看padding后修改的bbox数值是否正确，在原图中画出bbox检验
        # examine(img,bbox)
        labels = convert_labels(bbox)
        # 将所有bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)

        labels = torch.tensor(labels)
        return img, labels


def convert_labels(bbox, numbox=2, numclass=20):
    """
    将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式
    """
    gridsize = 1.0 / 7
    labels = np.zeros((7, 7, 5 * numbox + numclass))
    for i in range(len(bbox) // 5):
        gridx = int(bbox[i * 5 + 1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i * 5 + 2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx
        gridpy = bbox[i * 5 + 2] / gridsize - gridy

        # print(gridx,gridy)
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10 + int(bbox[i * 5])] = 1


    # print(labels[3][4])
    return labels

def examine(img,label):
    """检查dataset中对图像进行padding后x,y,w,h的正确性"""
    h, w = img.shape[:2]
    print(w, h)
    print(CLASSES[int(label[0])])
    pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))  # box左上角坐标
    pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))  # box右下角坐标
    cv2.putText(img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2))
    cv2.imshow("img", img)
    cv2.waitKey(0)

def get_data(batch_size=5):
    dataset = Voc2007(Dataset_path, "trainval", True)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader



if __name__ == "__main__":
    dataset= Voc2007(Dataset_path,"trainval",True)
    print(len(dataset))
    dataloader = DataLoader(dataset,32,shuffle=True)
    for img,target in dataloader:
        print("!",img.shape)
        print("!",target.shape)
        break
    for i in img:
        plt.imshow(i.numpy().transpose(1, 2, 0).astype(np.float32))
        plt.show()
        for k in range(1):
            for i in range(7):
                for j in range(7):
                    if torch.max(target[k][i][j]==1):
                        print(target[k][i][j])
        break