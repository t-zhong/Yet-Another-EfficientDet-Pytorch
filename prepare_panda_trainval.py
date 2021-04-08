# Author t-zhong
# Created 2020/3/27
# Modified 2020/4/7

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


def process_rect(region, rect, width, height, scale_ratio):
    """
    Determine whether the rect is in the region and convert rect values to relative values.
    region: [minw, minh, maxw, maxh], the coordinate of the top-left and bottom-right points of the splited image
    rect: original bounding box
    width: original image width
    height: original image height
    scale_ratio: resized image side length / image side length
    """
    minw, minh, maxw, maxh = region
    tl_x = width * rect['tl']['x']
    tl_y = height * rect['tl']['y']
    br_x = width * rect['br']['x']
    br_y = height * rect['br']['y']
    # Determine whether this object is in the segmented area.
    if minw <= tl_x < maxw and minh <= tl_y < maxh:
        if minw <= br_x < maxw and minh <= br_y < maxh:
            bbox = [tl_x - minw, tl_y - minh, br_x - tl_x, br_y - tl_y]
            bbox = [x * scale_ratio for x in bbox]
            return bbox
    return None


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


# def process_rect(width, height, rect)

# source
base_dir = os.path.expanduser('~/datasets/panda')
annos_dir = os.path.join(base_dir, 'panda_round1_train_annos_202104')
vehicle_path = os.path.join(annos_dir, 'vehicle_bbox_train.json')
person_path = os.path.join(annos_dir, 'person_bbox_train.json')
images_list = glob.glob(os.path.join(base_dir, 'panda_round1_train_202104_part?/*/*.jpg'))

# target
os.makedirs('datasets/panda/annotations', exist_ok=True)
os.makedirs('datasets/panda/train', exist_ok=True)
os.makedirs('datasets/panda/val', exist_ok=True)
target_base_dir = 'datasets/panda'

def images_generator(images_list):
    for image_path in images_list:
        yield image_path, cv2.imread(image_path)

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
# scale_ratio = 1.55
croped_image_id = -1
bbox_id = -1

# images_list = images_list[:10]

progress_bar = tqdm(images_generator(images_list), total=len(images_list))

for idx, (image_path, img) in enumerate(progress_bar):
    # if idx > 0:
    #     break
    # indicate whether croped images of this image is in train folder or in val folder
    group = 'val' if (idx + 1) % (1 / val_ratio) == 0 else 'train'
    short_path = '/'.join(image_path.split('/')[-2:])
    image_info = vehicle_bboxs[short_path]
    image_id = image_info['image id']
    image_size = image_info['image size']
    height = image_size['height']
    width = image_size['width']
    vehicle_objects = image_info['objects list']
    person_objects = person_bboxs[short_path]['objects list']

    side_length = 4608 # or very large number like 1e9 for whole image.
    overlap = 1536
    curh, curw = 0, 0
    while curh < height:
        while curw < width:
            # Determine the region
            minw = curw if curw + side_length <= width else width - side_length
            minh = curh if curh + side_length <= height else height - side_length
            maxw = min(curw + side_length, width)
            maxh = min(curh + side_length, height)
            
            # Process objects
            croped_image_id += 1
            contained_objs = []
            for obj in vehicle_objects:
                category = obj['category']
                if category in ['vehicles', 'unsure']:
                    continue
                rect = obj['rect']
                bbox = process_rect([minw, minh, maxw, maxh], rect, width, height, input_size / side_length)
                if bbox:
                    bbox_id += 1
                    contained_objs.append(gen_anno(croped_image_id, bbox, 4, bbox_id))
            
            for obj in person_objects:
                category = obj['category']
                if category != 'person':
                    continue
                rects = obj['rects']
                head = rects['head']
                visible_body = rects['visible body']
                full_body = rects['full body']

                bbox = process_rect([minw, minh, maxw, maxh], head, width, height, input_size / side_length)
                if bbox:
                    bbox_id += 1
                    contained_objs.append(gen_anno(croped_image_id, bbox, 1, bbox_id))

                bbox = process_rect([minw, minh, maxw, maxh], visible_body, width, height, input_size / side_length)
                if bbox:
                    bbox_id += 1
                    contained_objs.append(gen_anno(croped_image_id, bbox, 2, bbox_id))
                
                bbox = process_rect([minw, minh, maxw, maxh], full_body, width, height, input_size / side_length)
                if bbox:
                    bbox_id += 1
                    contained_objs.append(gen_anno(croped_image_id, bbox, 3, bbox_id))
            
            # print(contained_objs)
            
            # If there are objects in croped image, resize and then save this image.
            if len(contained_objs) > 0:
                croped_img = img[minh: maxh, minw: maxw, :]
                resized_img = cv2.resize(croped_img, (input_size, input_size))
                file_name = f'{image_id}_{minw}x{minh}_{side_length}_{input_size}.jpg'
                cv2.imwrite(os.path.join(target_base_dir, group, file_name), resized_img)
                
                croped_image_info = {'file_name': file_name, 'height': input_size, 'width': input_size, 'id': croped_image_id}
                if group == 'train':
                    train_images.append(croped_image_info)
                    train_annotations.extend(contained_objs)
                else:
                    val_images.append(croped_image_info)
                    val_annotations.extend(contained_objs)
            else:
                croped_image_id -= 1
                
            curw += side_length - overlap
        curh += side_length - overlap
        curw = 0
        # side_length = int(side_length * scale_ratio)
        # overlap = int(overlap * scale_ratio)

# categories
#   |-person
#       |-head
#       |-visible body
#       |-full body
#   |-fake person
#   |-ignore
#   |-crowd
#   |-vehcile
#       |-

categories = [
    {'supercategory': 'none', 'id': 1, 'name': 'head'},
    {'supercategory': 'none', 'id': 2, 'name': 'visible body'},
    {'supercategory': 'none', 'id': 3, 'name': 'full body'},
    {'supercategory': 'none', 'id': 4, 'name': 'visible car'}]

with open('datasets/panda/annotations/instances_train.json', 'w') as fp:
    json.dump({'images': train_images, 'annotations': train_annotations, 'categories': categories}, fp, indent=4)

with open('datasets/panda/annotations/instances_val.json', 'w') as fp:
    json.dump({'images': val_images, 'annotations': val_annotations, 'categories': categories}, fp, indent=4)


# simply test the above code

if False: 
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
