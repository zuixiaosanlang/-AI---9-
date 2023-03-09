
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.optimizer import lr
from model import PPLCNet
import random
import numpy as np
from paddle.optimizer import lr
from dataset import AngleClass
import matplotlib.pyplot as plt
import cv2

from configs.config_base import Config
train_dataset = AngleClass(Config, 'train', False)
img, label = train_dataset[100]
img = img.transpose([1,2,0])
# img, label = next(iter(iter(train_dataset)))
# print(img.shape,"Yes",img*255)
cv2.imwrite('/home/aistudio/CLS/code/2.jpg',img*255)
print("Save")
# img = img.transpose([1,2,0])
# plt.imshow(img)
# plt.show()
# test_dataset=AngleClass(Config, 'test', False)