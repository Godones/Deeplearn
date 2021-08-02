import os
import xml.etree.ElementTree as ET
import cv2

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
           'train', 'bottle', 'chair', 'dining table', 'potted plant',
           'sofa', 'tvmonitor']

Dataset_path = "./data/train/VOCdevkit/VOC2007/"


def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (float(x), float(y), float(w), float(h))


def convert_annotation(filename):
    """
        filename是图片对应的图片
        把图像image_id的xml文件转换为目标检测的label文件(txt)
        其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
        并将四个物理量归一化
    """
    imagefile = open(Dataset_path + "Annotations/{:}".format(filename))
    image_id = filename.split('.')[0]  # 获取图片的名称
    outfile = open('./labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(imagefile)
    root = tree.getroot()
    size = root.find('size')
    weight = int(size.find('width').text)  # 图片的宽度
    height = int(size.find('height').text)  # 图片的高度
    print(weight, height)
    # print(list(root.iter('object')))
    # print(list(list(list(root.iter('object'))[0])[-1]))
    for it in root.iter('object'):
        class_name = it.find('name').text
        if class_name not in CLASSES:
            continue
        class_name_id = CLASSES.index(class_name)  # 图片对应的类别

        xmlbox = it.find('bndbox')  # 找到box坐标
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                  float(xmlbox.find('ymax').text))
        box_process = convert((weight, height), points)
        # print(box_process)
        outfile.write(str(class_name_id) + " " + " ".join([str(a) for a in box_process]) + '\n')


def make_label_txt():
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    filenames = os.listdir(Dataset_path + 'Annotations')
    for file in filenames:
        convert_annotation(file)


def show_labels_img(imgname):
    """imgname：图片名称"""
    img = cv2.imread(Dataset_path + "JPEGImages/" + imgname + ".jpg")
    h, w = img.shape[:2]
    print(w, h)
    label = []
    with open("./labels/" + imgname + ".txt", 'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(CLASSES[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))#box左上角坐标
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))#box右下角坐标
            cv2.putText(img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2))

    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    print(os.getcwd())
    if not os.path.exists("./labels"):
        make_label_txt()
    show_labels_img("000005")
