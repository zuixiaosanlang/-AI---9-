import paddle
from paddle.vision.transforms import to_tensor
from paddle.io import Dataset
import matplotlib.pyplot as plt
import cv2
import os
import json
import math
import zipfile
import random
import numpy as np
from PIL import Image, ImageEnhance
from transform import rotate_image, distort_color, ResizePad,  cutOut, RandomErasing, rotate_angle

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import copy
import urllib
from math import *


def rotate_img_bbox(img, bboxes, angle=45, scale=1.):


    # 角度变弧度
    height, width = img.shape[:-1]
    degree=angle
    height_new = int(width*abs(sin(radians(degree)))+height*abs(cos(radians(degree))))
    width_new = int(height*abs(sin(radians(degree)))+width*abs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0,2]+=(width_new-width)/2
    matRotation[1,2]+=(height_new-height)/2
    imgRotation = cv2.warpAffine(img, M=matRotation,dsize=(width_new,height_new),borderValue=(255,255,255))
    

    #---------------------- 矫正bbox坐标 ----------------------
    # rot_mat是最终的旋转矩阵
    rot_bboxes = list()

    bbox = np.array(bboxes).reshape([-1,2])

    point1 = np.dot(matRotation, np.array([bbox[0][0], bbox[0][1], 1]).astype(np.int32))
    point2 = np.dot(matRotation, np.array([bbox[1][0], bbox[1][1], 1]).astype(np.int32))
    point3 = np.dot(matRotation, np.array([bbox[2][0], bbox[2][1], 1]).astype(np.int32))
    point4 = np.dot(matRotation, np.array([bbox[3][0], bbox[3][1], 1]).astype(np.int32))

    # 加入list中
    rot_bboxes.append([[point1[0], point1[1]], 
                        [point2[0], point2[1]], 
                        [point3[0], point3[1]], 
                        [point4[0], point4[1]]])
    return imgRotation, rot_bboxes

def random_crop(img, size, scale=[0.3, 1.2], ratio =[3./4., 4./3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. /aspect_ratio
    bound = min((float(img.shape[0])/img.shape[1]) / (w**2),
                               (float(img.shape[1])/img.shape[0])/(h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_areas = img.shape[0] * img.shape[1]* np.random.uniform(scale_min, scale_max)
    target_size = math.sqrt(target_areas)
    w = int(target_size* w)
    h = int(target_size*h)

    i = np.random.randint(0, img.shape[0] - w +1)
    j = np.random.randint(0, img.shape[1]- h +1)
    img = img[j:j+h,i:i+w]
    img = cv2.resize(img,(size,size), Image.LANCZOS)
    return img

def cutOut(img):
    n_holes =  np.random.randint(0, 5)
    h, w = img.shape[:2]
    length_list= [16, 32, 64]
    i = np.random.randint(0, len(length_list))
    length = length_list[i]
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y-length//2, 0, h)
        y2 = np.clip(y+length//2, 0, h)
        x1 = np.clip(x-length//2, 0, w)
        x2 = np.clip(x+length//2, 0, w)
        img[y1:y2, x1:x2] =0
    return img

class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        # print(self.std,self.scale,self.mean)
        img = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return img


def process_image(img, mode, rotate, resize_with_pad):
    if mode == 'train':
        img = img.convert('RGB')
        img = distort_color(img)
        img = np.array(img)
        img = ResizePad(img, target_size=resize_with_pad)
        
        # img = random_crop(img,600)
        # prob = np.random.uniform(0, 1)
        # # if prob<0.5:
        #     img = cutOut(img)

    else:
        img = np.array(img)
        img = ResizePad(img, target_size=624)
        # img = crop_image(img, 600,True)
    img = np.array(img).astype('float32').transpose((2, 0, 1)) /255.0
#     norm= NormalizeImage()
#     img = norm(img)

    return img

def crop_image(img, target_size, center):
    width, height = img.shape[:-1]
    size = target_size
    if center == True:
        w_start = (width-size) /2
        h_start = (height-size)/2
    else:
        w_start = np.random.randint(0, width-size+1)
        h_start = np.random.randint(0, height-size+1)
    w_end = w_start+size
    h_end = h_start + size
    img = img[int(h_start):int(h_end),int(w_start):int(w_end),:]
    return img

def get_rotate_label(boxes):

    label = 0
    box = np.array(boxes).reshape([-1,2])
    x_c,y_c = box[0,0],box[0,1]
    x = box[:,0]
    l_idx = x.argsort()
    l_box = np.array([box[l_idx[0]],box[l_idx[1]]])
    r_box = np.array([box[l_idx[2]],box[l_idx[3]]])
    l_idx_1 = np.array(l_box[:,1]).argsort()
    lt = l_box[l_idx_1[0]]
    lb = l_box[l_idx_1[1]]
    r_idx_1 = np.array(r_box[:,1]).argsort()
    rt = r_box[r_idx_1[0]]
    rb = r_box[r_idx_1[1]]
    set_b = [lt, rt, rb, lb]
    for idx, (x1,y1) in enumerate(set_b):
        if x1 == x_c and y1==y_c:
            label = idx
            break
    return label

def expand():
    expand_ratio =  list(np.arange(0, 50, 10))
    expand_i = np.random.randint(0,len(expand_ratio))
    expand_ymin = expand_ratio[expand_i]
    return expand_ymin 

def get_expand_dst():
    xmin = expand()
    ymin = expand()
    xmax = expand()
    ymax = expand()
    return xmin,ymin, xmax,ymax

class AngleClass1(Dataset):

    def __init__(self, Config, mode, isFine):
        base_dataset_path =Config["dataset"]["base_dataset_path"]
        base_train_label_path = Config["dataset"]["base_train_label_path"]
        base_test_label_path = Config["dataset"]["base_test_label_path"]

        self.train_img, self.train_label, self.mode, self.aug_img, self.aug_label = [], [], mode, [], []
        self.train_bbox = []
        self.resize_with_pad = Config["dataset"]["resize_pad"]
        self.isFineTuning = False     
        if mode=='train':
            load_p = base_train_label_path
        else:
            load_p = base_test_label_path   

        with open(load_p,'r') as f:
            lines = f.readlines()
        for (lid,line) in enumerate(lines): 
            img_name, box = line.split('\t')

            if not img_name.endswith(".jpg"):
                continue
            file_path = os.path.join(base_dataset_path, img_name)
            img = Image.open(file_path)
           
            if img is not None:
                
                boxes = json.loads(box)

                self.train_img.append(file_path)
                self.train_label.append(0)
                self.train_bbox.append(boxes)

             
    def __len__(self):
        return len(self.train_img)
    def __getitem__(self,index):

        img_path = self.train_img[index]
        img = Image.open(img_path)
        box = self.train_bbox[index]
        box = np.array(box).reshape([-1,2])
        img = np.array(img)
        h1,w1 = img.shape[:-1]
        xmin = min(box[:,0])
        xmax = max(box[:,0])
        ymin = min(box[:,1])
        ymax = max(box[:,1])
        # xmin_e,ymin_e, xmax_e,ymax_e = get_expand_dst()

        # ymin_a = max(ymin-ymin_e,0)
        # xmin_a = max(xmin-xmin_e,0)
        # ymax_a = min(ymax+ymax_e, h1)
        # xmax_a = min(xmax+xmax_e,w1)
        img = img[int(ymin):int(ymax),int(xmin):int(xmax),:] 
        box[:,0] = box[:,0]-xmin
        box[:,1] = box[:,1]-ymin 

        label_list = np.arange(0,4,1)
        label_i = np.random.randint(0,len(label_list))
        label_a = label_list[label_i]
        # print(label_a )
        if label_a ==0:
            angle_list = np.arange(0,90,10)
        elif label_a ==1:
            angle_list = np.arange(90,180,10)
        elif label_a ==2:
            angle_list = np.arange(180,270,10)
        else:
            angle_list = np.arange(270,360,10)
        angle_i = np.random.randint(0,len(angle_list))
        angle = angle_list[angle_i]
        # print(angle)
        img,box = rotate_img_bbox(img,box,angle=angle, scale=0.5) 
        cv2.polylines(img, [np.array(box).astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 0, 255), thickness=10)    
        label = get_rotate_label(box)


        img =  Image.fromarray(img)

        img = process_image(img, self.mode, True, self.resize_with_pad)
        
        label = paddle.to_tensor(int(label))
        
        return img, label

# # 768 512
# train_dataset = AngleClass(Config, 'train')
# train_loader = paddle.io.DataLoader(train_dataset, batch_size = Config["dataset"]["train_batch"], shuffle=True, num_workers=0)

# test_dataset=AngleClass(Config, 'test')
# test_loader = paddle.io.DataLoader(test_dataset,batch_size = Config["dataset"]["test_batch"], shuffle=False,num_workers=0)


# train_img, label = next(iter(train_loader))
# test_img, label1 = next(iter(test_loader))