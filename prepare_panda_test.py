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

# source
base_dir = os.path.expanduser('~/datasets/panda')
annos_dir = os.path.join(base_dir, 'panda_round1_test_A_annos_202104')
vehicle_path = os.path.join(annos_dir, 'vehicle_bbox_test_A.json')
person_path = os.path.join(annos_dir, 'person_bbox_test_A.json')
images_list = glob.glob(os.path.join(base_dir, 'panda_round1_test_202104_A/*/*.jpg'))

# target
target_base_dir = 'datasets/panda'
os.makedirs('datasets/panda/annotations', exist_ok=True)
os.makedirs('datasets/panda/test', exist_ok=True)


def images_generator(images_list):
    for image_path in images_list:
        yield image_path, cv2.imread(image_path)

with open(vehicle_path) as fp:
    vehicle_bboxs = json.load(fp)

with open(person_path) as fp:
    person_bboxs = json.load(fp)

test_images = []

compound_coef = 0
input_sizes = [512, 512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] # model input size
# scale_ratio = 1.55
croped_image_id = -1
bbox_id = -1

# images_list = images_list[:1]

progress_bar = tqdm(images_generator(images_list), total=len(images_list))

for idx, (image_path, img) in enumerate(progress_bar):
    # if idx > 0:
    #     break
    # indicate whether croped images of this image is in train folder or in val folder
    short_path = '/'.join(image_path.split('/')[-2:])
    image_info = vehicle_bboxs[short_path]
    image_id = image_info['image id']
    image_size = image_info['image size']
    height = image_size['height']
    width = image_size['width']

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
            
            croped_img = img[minh: maxh, minw: maxw, :]
            resized_img = cv2.resize(croped_img, (input_size, input_size))
            file_name = f'{image_id}_{minw}x{minh}_{side_length}_{input_size}.jpg'
            cv2.imwrite(os.path.join(target_base_dir, 'test', file_name), resized_img)
            
            croped_image_info = {'file_name': file_name, 'height': input_size, 'width': input_size, 'id': croped_image_id}
            test_images.append(croped_image_info)
                
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

with open('datasets/panda/annotations/image_info_test.json', 'w') as fp:
    json.dump({'images': test_images, 'annotations': [], 'categories': categories}, fp, indent=4)

