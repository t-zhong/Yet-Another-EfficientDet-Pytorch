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
from pycocotools.coco import COCO


def cvt_bbox(minw, minh, side_length, target_length, bbox):
    """
    Convert original bounding box values to relative values.
    tl_x, tl_y: the coordinate of the top left point of the splited image
    slide_length: the side length of the splited image
    target_length: the side length of the resized image
    bbox: [tl_x, tl_y, bt_x, bt_y] 
    """

    bbox = [bbox[0] - minw, bbox[1] - minh, bbox[2] - bbox[0], bbox[3] - bbox[1]]
    scale_ratio = target_length / side_length
    bbox = [x * scale_ratio for x in bbox]
    return bbox


def gen_anno(image_id, bbox, category_id, id, area=0, iscrowd=0, ignore=0, segmentation=[]):
    """
    generate a annotation like coco-style
    """
    return {
        'area': area,
        'iscrowd': iscrowd,
        'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id, 
        'id': id,
        'ignore': ignore,
        'segmentation': segmentation,
    }

def plot_bboxs(img, bboxs):
    """
    bboxs: list of bounding boxs, [x, y, w, h]
    """
    for bbox in bboxs:
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]))
    return img

base_dir = os.path.expanduser('~/datasets/panda')
annos_dir = os.path.join(base_dir, 'panda_round1_train_annos_202104')
vehicle_path = os.path.join(annos_dir, 'vehicle_bbox_train.json')
person_path = os.path.join(annos_dir, 'person_bbox_train.json')
images_dir1 = os.path.join(base_dir, 'panda_round1_train_202104_part1')
images_dir2 = os.path.join(base_dir, 'panda_round1_train_202104_part2')

os.makedirs('datasets/panda_vehicle/annotations', exist_ok=True)
os.makedirs('datasets/panda_vehicle/train', exist_ok=True)
os.makedirs('datasets/panda_vehicle/val', exist_ok=True)

target_base_dir = 'datasets/panda_vehicle'

# def images_generator(images_list):
#     for image_path in images_list:
#         yield image_path, cv2.imread(image_path)
#         # yield image_path, np.zeros(1)

with open(vehicle_path) as fp:
    vehicle_bboxs = json.load(fp)

with open(person_path) as fp:
    person_bboxs = json.load(fp)

# def filt_bboxs(minh, minw, maxh, maxw, height, width, objects_list):
#     pass

val_ratio = 0.1

train_images = []
val_images = []
# list contains annotations
train_annotations = []
val_annotations = []

compound_coef = 0
input_sizes = [512, 512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] # model input size
scale_ratio = 1.55
splited_image_id = -1
bbox_id = -1

for idx, (image_path, image_info) in enumerate(vehicle_bboxs.items()):
    if idx > 0:
        break
    # indicate whether croped images of this image is in train folder or in val folder
    group = 'val' if (idx + 1) % (1 / val_ratio) == 0 else 'train'
    if int(image_path.split('_')[0]) <= 8:
        image_path = os.path.join(images_dir1, image_path)
    else:
        image_path = os.path.join(images_dir2, image_path)
    img = cv2.imread(image_path)
    image_id = image_info['image id']
    image_size = image_info['image size']
    height = image_size['height']
    width = image_size['width']
    objects_list = image_info['objects list']

    side_length = 4608 # or very large number like 1e9 for whole image.
    overlap = 1536
    curh, curw = 0, 0
    while curh < height:
        while curw < width:
            minw = curw
            minh = curh
            maxw = min(curw + side_length, width)
            maxh = min(curh + side_length, height)
            
            splited_image_id += 1
            contained_objs = []
            for obj in objects_list:
                category = obj['category']
                rect = obj['rect']
                tl_x = width * rect['tl']['x']
                tl_y = height * rect['tl']['y']
                br_x = width * rect['br']['x']
                br_y = height * rect['br']['y']
                # Determine whether this object is in the segmented area.
                if minw <= tl_x < maxw and minh <= tl_y < maxh:
                    if minw <= br_x < maxw and minh <= br_y < maxh:
                        bbox_id += 1
                        bbox = cvt_bbox(minw, minh, side_length, input_size, [tl_x, tl_y, br_x, br_y])
                        contained_objs.append(gen_anno(splited_image_id, bbox, 1, bbox_id))
            
            if len(contained_objs) > 0:
                # If there are objects in the splited image, resize and then save this image.
                croped_img = img[minh: maxh, minw: maxw, :]
                resized_img = cv2.resize(croped_img, (input_size, input_size))
                file_name = f'{image_id}_{minw}x{minh}_{side_length}_{input_size}.jpg'
                cv2.imwrite(os.path.join(target_base_dir, group, file_name), resized_img)
                
                croped_image_info = {'file_name': file_name, 'height': input_size, 'width': input_size, 'id': splited_image_id}
                if group == 'train':
                    train_images.append(croped_image_info)
                    train_annotations.append(contained_objs)
                else:
                    val_images.append(croped_image_info)
                    val_annotations.append(contained_objs)
                
            curw += side_length - overlap
        curh += side_length - overlap
        curw = 0
        # side_length = int(side_length * scale_ratio)
        # overlap = int(overlap * scale_ratio)


categories = [{'supercategory': 'none', 'id': 1, 'name': 'visible car'}]
with open('datasets/panda_vehicle/annotations/instances_train.json', 'w') as fp:
    json.dump({'images': train_images, 'annotations': train_annotations, 'categories': categories}, fp)

with open('datasets/panda_vehicle/annotations/instances_val.json', 'w') as fp:
    json.dump({'images': val_images, 'annotations': val_annotations, 'categories': categories}, fp)
        
cocoapi = COCO('datasets/panda_vehicle/annotations/instances_train.json')
# def in_region(obj, region):
#     """
#     Determine whether an object is in the given croped region.
#     obj: [x, y, w, h] region: [x1, y1, x2, y2]
#     """
#     x, y, w, h = obj
#     x1, y1, x2, y2 = region
#     if x1 <= x < x2 and y1 <= y < y2:
#         if x + w < x2 and y + h < y2:
#             return True
#     return False


# def shift_bbox(tl_x, tl_y, bbox):
#     """
#     bbox: [x, y, w, h]
#     """

#     return [
#         bbox[0] - tl_x,
#         bbox[1] - tl_y,
#         bbox[2],
#         bbox[3],
#     ]

# def resize(img, bndboxs):
#     """
#     Resize the image and convert the bounding boxs.
#     """
#     h, w = img.shape
#     scale_ratio = 1536 / w
#     img = cv2.resize(img, 1536, h / w * 1536)
#     scaled_bndboxs = []
#     for bndbox in bndboxs:
#         scaled_bndbox = [b * scale_ratio for b in bndbox]
#         scaled_bndboxs.append(scaled_bndbox)
#     return img, scaled_bndboxs
    

# images_list = images_list
# progress_bar = tqdm(images_generator(images_list), total=len(images_list))
# for idx, (image_path, img) in enumerate(progress_bar):
#     split = 'val' if (idx + 1) % int(1 / val_ratio) == 0 else 'train'
#     # img = cv2.imread(image_path)
#     # Getting the dict key in the annotation.
#     dict_key = '/'.join(image_path.split('/')[-2:])
#     vehicle_dict = vehicle_bboxs[dict_key]
#     image_size = vehicle_dict['image size']
#     height, width = image_size['height'], image_size['width']
#     image_id = vehicle_dict['image id']
#     vehicle_objects = vehicle_dict['objects list']

#     file_name = f'{image_id}_0x0.jpg'
#     # cv2.imwrite(f'datasets/panda_vehicle/{split}/{file_name}', img)
#     # shutil.copy(image_path, f'datasets/panda_vehicle/{split}/{file_name}')
#     image_info = {'file_name': file_name,
#                   'height': height, 'width': width, 'id': image_id}
#     if split == 'train':
#         train_images.append(image_info)
#     else:
#         val_images.append(image_info)

#     all_objs = []
#     for obj in vehicle_objects:
#         if obj['category'] not in ['vehicles', 'unsure']:
#             all_objs.append({'category': 'visible car',
#                             'rect': cvt_bbox(height, width, obj['rect']), })

#     for obj in all_objs:
#         id += 1
#         anno = {'id': id, 'image_id': image_id, 'category_id': 1,
#                 'segmentation': [], 'area': 0,  'bbox': obj['rect'], 'iscrowd': 0}
#         if split == 'train':
#             train_annotations.append(anno)
#         else:
#             val_annotations.append(anno)


