import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fastercnn
from matplotlib import pyplot as plt
import cv2
import numpy as np
from torchvision.datasets import CocoCaptions
import os
from PIL import Image


def show_rect(image: np.ndarray, bndbox):
    pt1 = bndbox[:2]
    pt2 = bndbox[2:]
    image_show = image
    return cv2.rectangle(image_show, pt1, pt2, (0, 255, 255), 2)

def show_name(image, name, p_tl):
    return cv2.putText(image, name, p_tl, 1, 1, (255, 255, 0))



def image_load(fileos):
    images = cv2.imread(fileos)
    orinimg = images
    images = images.transpose(2,0,1)
    images = np.expand_dims(images,axis=0)
    print(images.shape)
    images =torch.tensor(images.astype(np.float32)/255.0)
    # xmin = prediction[0]["boxes"][0][0]
    # ymin = prediction[0]['boxes'][0][1]
    # xmax = prediction[0]["boxes"][0][2]
    # ymax = prediction[0]['boxes'][0][3]
    # name = "car: " + str(prediction[0]['scores'][0])
    return images

def errors(image):
    image_patch = torch.rand_like(image,requires_grad=True)
    image_add = torch.zeros_like(image,requires_grad=True)
    # print(image_patch)
    image.requires_grad = True

    Fastmodel = fastercnn(pretrained=True)
    for param in Fastmodel.parameters():
        param.requires_grad = False
    Fastmodel.eval()

    predict = Fastmodel(image)
    print(predict)
    myanswer = [{"boxes":predict[0]["boxes"],"labels":predict[0]['labels'],
                 "scores":predict[0]['scores']}]
    myanswer[0]['labels'] = torch.tensor([19])
    print(myanswer)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([image_patch],lr=0.1)

    # Fastmodel.to(device="cuda")
    for epoch in range(100):
        image_add = image + image_patch
        # image_add = image_add.to("cuda")
        predict_ = Fastmodel(image_add)
        # print(predict_)
        indexmax = torch.argmax(predict_[0]['scores'])
        if predict_[0]['labels'][indexmax]==myanswer[0]['labels'][0]:
            print("sss")
            break
        losses = loss(predict_[0]['boxes'][indexmax],myanswer[0]['boxes'][0])+loss(
            predict_[0]['labels'][indexmax]*1.0,myanswer[0]['labels'][0]*1.0
        )
        print(losses)
        optimizer.zero_grad()
        losses.backward()


if __name__ == "__main__":
    file = "car1.jpg"
    image = image_load(file)
    print(image.size())
    # errors(image)
    labels = torch.randint(1, 91, (4, 11))
    print(labels.size())
    errors(image)


