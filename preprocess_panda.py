# Author t-zhong

"""
Simple script for calculating max. width and height of bbox.
"""

import json
import os

import yaml


base_dir = '/root/datasets/panda'
annos_dir = os.path.join(base_dir, 'panda_round1_train_annos_202104')
vehicle_path = os.path.join(annos_dir, 'vehicle_bbox_train.json')
person_path = os.path.join(annos_dir, 'person_bbox_train.json')

with open(vehicle_path) as fp:
    vehicle_bboxs = json.load(fp)

with open(person_path) as fp:
    person_bboxs = json.load(fp)

middle_xs, middle_ys = [], []
max_w, max_h = .0, .0
min_w, min_h = 1e9, 1e9

for image_path, values in vehicle_bboxs.items():
    image_id = values['image id']
    image_size = values['image size']
    height, width = image_size['height'], image_size['width']
    for obj in values['objects list']:
        category = obj['category']
        rect = obj['rect']
        tl_x, tl_y = rect['tl']['x'], rect['tl']['y']
        br_x, br_y = rect['br']['x'], rect['br']['y']
        middle_xs.append((tl_x + br_x) / 2)
        middle_ys.append((tl_y + br_y) / 2)
        bbox_w, bbox_h = (br_x - tl_x) * width, (br_y - tl_y) * height
        max_w, max_h = max(max_w, bbox_w), max(max_h, bbox_h)
        min_w, min_h = min(min_w, bbox_w), min(min_h, bbox_h)
        

print(f'max width of vehicle: {max_w}, max height of vehicle: {max_h}')
print(f'min width of vehicle: {min_w}, min height of vehicle: {min_h}')

max_w, max_h = .0, .0
min_w, min_h = 1e9, 1e9

for image_path, values in person_bboxs.items():
    image_id = values['image id']
    image_size = values['image size']
    height, width = image_size['height'], image_size['width']
    for obj in values['objects list']:
        category = obj['category']
        if category == 'person':
            rects = obj['rects']
            head = rects['head']
            visible_body = rects['visible body']
            full_body = rects['full body']
            head_w = head['br']['x'] - head['tl']['x']
            head_h = head['br']['y'] - head['tl']['y']
            visible_body_w = visible_body['br']['x'] - visible_body['tl']['x']
            visible_body_h = visible_body['br']['y'] - visible_body['tl']['y']
            full_body_w = full_body['br']['x'] - full_body['tl']['x']
            full_body_h = full_body['br']['y'] - full_body['tl']['y']
            max_w = max(max_w, max(head_w, visible_body_w, full_body_w) * width)
            max_h = max(max_h, max(head_h, visible_body_h, full_body_h) * height)
            min_w = min(min_w, min(head_w, visible_body_w, full_body_w) * width)
            min_h = min(min_h, min(head_h, visible_body_h, full_body_h) * height)

print(f'max width of person bbox: {max_w}, max height of person bbox: {max_h}')
print(f'min width of person bbox: {min_w}, min height of person bbox: {min_h}')
            