import paddle
from paddle.vision.transforms import to_tensor
from paddle.io import Dataset
import matplotlib.pyplot as plt
import cv2
import os
import math
import zipfile
import random
import numpy as np
from PIL import Image, ImageEnhance

train_parameters = {
    "input_size": [3, 224, 224],
    "class_dim": -1,  # 分类数，会在初始化自定义 reader 的时候获得
    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得
    "label_dict": {},
    "data_dir": "data/data2815",  # 训练数据存储地址
    "train_file_list": "train.txt",
    "eval_file_list": "eval.txt",
    "label_file": "label_list.txt",
    "continue_train": False,  # 是否接着上一次保存的参数接着训练
    "mode": "train",
    "num_epochs": 15,
    "layer": 16,
    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值
    "image_enhance_strategy": {  # 图像增强相关策略
        "need_distort": True,  # 是否启用图像颜色增强
        "need_rotate": True,  # 是否需要增加随机角度
        "need_crop": True,  # 是否要增加裁剪
        "need_flip": True,  # 是否要增加水平随机翻转
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },

}


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resize_width = int(round(img.size[0] * percent))
    resize_height = int(round(img.size[1] * percent))
    img = img.resize((resize_width, resize_width), Image.LANCZOS)
    return img


def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio
    bound = min((float(img.size[0]) / img.size[1]) / (w ** 2),
                (float(img.size[1]) / img.size[0]) / (h ** 2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_areas = img.size[0] * img.size[1] * np.random.uniform(scale_min, scale_max)
    target_size = math.sqrt(target_areas)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)
    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def random_brightness(img):
    """
    图像增强，亮度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_enhance_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    """
    图像增强，对比度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_enhance_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    """
    图像增强，饱和度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_enhance_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    """
    图像增强，色度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['hue_prob']:
        hue_delta = train_parameters['image_enhance_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_color(img):
    """
    概率的图像增强
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob < 0.5:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    elif prob < 0.5:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def rotate_image(img):
    angle = np.random.randint(-5, 5)
    img = img.rotate(angle)
    return img


def rotate_angle(img, gt):
    i = np.random.randint(0, 4)
    angle = [100, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
    angle_choose = angle[i]  # [0,0,1,2]
    if angle_choose != 100:
        img = img.transpose(angle_choose)
        gt = int(gt) - i
        if gt < 0:
            gt = gt + 4
    return img, gt, i


def ResizePad(img, target_size):
    h, w = img.shape[:2]
    m = max(h, w)
    ratio = target_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    top = (target_size - new_h) // 2
    bottom = (target_size - new_h) - top
    left = (target_size - new_w) // 2
    right = (target_size - new_w) - left
    # 1img1 = cv2.copyMakeBorder(img, top, bottom,left, right,1)
    img1 = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # print("border_img",img1)
    return img1


def cutOut(img):
    n_holes = 1
    h, w = img.shape[:2]
    length_list = [16, 32, 64, 96]
    i = np.random.randint(0, 4)
    length = length_list[i]
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        img[y1:y2, x1:x2] = 0
    return img


class Pixels(object):
    def __init__(self, mode="const", mean=[0., 0., 0.]):
        self._mode = mode
        self._mean = np.array(mean)

    def __call__(self, h=224, w=224, c=3, channel_first=False):
        if self._mode == "rand":
            return np.random.normal(size=(
                1, 1, 3)) if not channel_first else np.random.normal(size=(
                3, 1, 1))
        elif self._mode == "pixel":
            return np.random.normal(size=(
                h, w, c)) if not channel_first else np.random.normal(size=(
                c, h, w))
        elif self._mode == "const":
            return np.reshape(self._mean, (
                1, 1, c)) if not channel_first else np.reshape(self._mean,
                                                               (c, 1, 1))
        else:
            raise Exception(
                "Invalid mode in RandomErasing, only support \"const\", \"rand\", \"pixel\""
            )


class RandomErasing(object):
    """RandomErasing.
    """

    def __init__(self,
                 EPSILON=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=[0., 0., 0.],
                 attempt=100,
                 use_log_aspect=False,
                 mode='const'):
        self.EPSILON = eval(EPSILON) if isinstance(EPSILON, str) else EPSILON
        self.sl = eval(sl) if isinstance(sl, str) else sl
        self.sh = eval(sh) if isinstance(sh, str) else sh
        r1 = eval(r1) if isinstance(r1, str) else r1
        self.r1 = (math.log(r1), math.log(1 / r1)) if use_log_aspect else (
            r1, 1 / r1)
        self.use_log_aspect = use_log_aspect
        self.attempt = attempt
        self.get_pixels = Pixels(mode, mean)

    def __call__(self, img):
        if random.random() > self.EPSILON:
            return img

        for _ in range(self.attempt):
            if isinstance(img, np.ndarray):
                img_h, img_w, img_c = img.shape
                channel_first = False
            else:
                img_c, img_h, img_w = img.shape
                channel_first = True
            area = img_h * img_w

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(*self.r1)
            if self.use_log_aspect:
                aspect_ratio = math.exp(aspect_ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                pixels = self.get_pixels(h, w, img_c, channel_first)
                x1 = random.randint(0, img_h - h)
                y1 = random.randint(0, img_w - w)
                if img_c == 3:
                    if channel_first:
                        img[:, x1:x1 + h, y1:y1 + w] = pixels
                    else:
                        img[x1:x1 + h, y1:y1 + w, :] = pixels
                else:
                    if channel_first:
                        img[0, x1:x1 + h, y1:y1 + w] = pixels[0]
                    else:
                        img[x1:x1 + h, y1:y1 + w, 0] = pixels[:, :, 0]
                return img
        return img
