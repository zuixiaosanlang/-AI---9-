import os
import cv2
from tqdm import tqdm
import json


img_root = '/home/aistudio/work/train/imgs'
label_p = '/home/aistudio/work/PPLCNet/label.txt'

with open('/home/aistudio/work/train/annos.txt','r',encoding='utf8')as fp:
    json_data = json.load(fp)
tbar = tqdm(json_data)
with open(label_p,'w') as f:
    for key in tbar :
        box_list = json_data[key]
        image_path = os.path.join(img_root,key)
        img = cv2.imread(image_path)
        h, w = img.shape[:-1]
        for idx1, item in enumerate(box_list):
            box = item['box']
            
            lb = item['lb']
            lt = item['lt']
            rt = item['rt']
            rb = item['rb']
            boxes = lt + rt + rb + lb
            box_w = json.dumps(boxes, ensure_ascii=False).replace('"', '').replace('\\', '"')

            filename = os.path.basename(image_path)
            f.writelines( '{}\t{}\n'.format(filename, box_w) )

    f.close()
