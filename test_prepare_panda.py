import glob
import json
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

cocoapi = COCO('datasets/panda/annotations/instances_train.json')

image_info = cocoapi.loadImgs(ids=[6])[0]
img = cv2.imread(os.path.join('datasets/panda/train', image_info['file_name']))
anns = cocoapi.loadAnns(cocoapi.getAnnIds(6))
# print(anns)


for ann in anns:
    bbox = ann['bbox']
    bbox = [int(x) for x in bbox]
    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1)
cv2.imwrite(image_info['file_name'], img)