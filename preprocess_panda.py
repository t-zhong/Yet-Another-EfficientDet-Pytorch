# Author t-zhong
# Created 2020/3/27

"""
Simple script for splitting original images and converting anno to coco-style.
"""
import glob
import json
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def cvt_bbox(height, width, bbox):
    """
    Convert original annotation to coco style annotation.
    """
    return [
        round(bbox['tl']['x'] * width, 2),
        round(bbox['tl']['y'] * height, 2),
        round((bbox['br']['x'] - bbox['tl']['x']) * width, 2),
        round((bbox['br']['y'] - bbox['tl']['y']) * height, 2)
    ]


def in_region(obj, region):
    """
    Determine whether an object is in the given croped region.
    obj: [x, y, w, h] region: [x1, y1, x2, y2]
    """
    x, y, w, h = obj
    x1, y1, x2, y2 = region
    if x1 <= x < x2 and y1 <= y < y2:
        if x + w < x2 and y + h < y2:
            return True
    return False


def shift_bbox(tl_x, tl_y, bbox):
    return [
        bbox[0] - tl_x,
        bbox[1] - tl_y,
        bbox[2],
        bbox[3],
    ]


base_dir = os.path.expanduser('~/datasets/panda')
annos_dir = os.path.join(base_dir, 'panda_round1_train_annos_202104')
vehicle_path = os.path.join(annos_dir, 'vehicle_bbox_train.json')
person_path = os.path.join(annos_dir, 'person_bbox_train.json')

with open(vehicle_path) as fp:
    vehicle_bboxs = json.load(fp)

with open(person_path) as fp:
    person_bboxs = json.load(fp)

images_list = glob.glob(os.path.join(
    base_dir, 'panda_round1_train_202104_part?/*/*.jpg'))

# Preprocess vehicle objects.
os.makedirs('datasets/panda_vehicle/annotations', exist_ok=True)
os.makedirs('datasets/panda_vehicle/train', exist_ok=True)
os.makedirs('datasets/panda_vehicle/val', exist_ok=True)

val_ratio = 0.1

train_images = []
val_images = []
# list contains annotations
train_annotations = []
val_annotations = []
# id assigned to each bounding box
id = -1


def images_generator(images_list):
    for image_path in images_list:
        # yield image_path, cv2.imread(image_path)
        yield image_path, np.zeros(1)


images_list = images_list
progress_bar = tqdm(images_generator(images_list), total=len(images_list))
for index, (image_path, img) in enumerate(progress_bar):
    split = 'val' if (index + 1) % int(1 / val_ratio) == 0 else 'train'
    # img = cv2.imread(image_path)
    # Getting the dict key in the annotation.
    dict_key = '/'.join(image_path.split('/')[-2:])
    vehicle_dict = vehicle_bboxs[dict_key]
    image_size = vehicle_dict['image size']
    height, width = image_size['height'], image_size['width']
    image_id = vehicle_dict['image id']
    vehicle_objects = vehicle_dict['objects list']

    file_name = f'{image_id}_0x0.jpg'
    # cv2.imwrite(f'datasets/panda_vehicle/{split}/{file_name}', img)
    shutil.copy(image_path, f'datasets/panda_vehicle/{split}/{file_name}')
    image_info = {'file_name': file_name,
                  'height': height, 'width': width, 'id': image_id}
    if split == 'train':
        train_images.append(image_info)
    else:
        val_images.append(image_info)

    all_objs = []
    for obj in vehicle_objects:
        if obj['category'] not in ['vehicles', 'unsure']:
            all_objs.append({'category': 'visible car',
                            'rect': cvt_bbox(height, width, obj['rect']), })

    for obj in all_objs:
        id += 1
        anno = {'id': id, 'image_id': image_id, 'category_id': 1,
                'segmentation': [], 'area': 0,  'bbox': obj['rect'], 'iscrowd': 0}
        if split == 'train':
            train_annotations.append(anno)
        else:
            val_annotations.append(anno)

categories = [{'supercategory': 'none', 'id': 1, 'name': 'visible car'}]
with open('datasets/panda_vehicle/annotations/instances_train.json', 'w') as fp:
    json.dump({'images': train_images, 'type': 'instances',
              'annotations': train_annotations, 'categories': categories}, fp)

with open('datasets/panda_vehicle/annotations/instances_val.json', 'w') as fp:
    json.dump({'images': val_images, 'type': 'instances',
              'annotations': val_annotations, 'categories': categories}, fp)
