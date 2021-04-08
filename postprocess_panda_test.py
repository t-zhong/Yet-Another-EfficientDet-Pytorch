# author: t-zhong
# Created: 2020/4/8

"""
postprocess test results,
"""

import json

from pycocotools.coco import COCO

project = 'panda'
checkpoint = 'efficientdet-d0_274_84668.pth'

cocoapi = COCO('datasets/panda/annotations/image_info_test.json')
# print(cocoapi.loadImgs(2))

with open(f'results/{project}/{checkpoint}/results.json', 'r') as fp:
    meta_results = json.load(fp)

final_results = []

for meta_result in meta_results:
    meta_image_id = meta_result['image_id']
    image_info = cocoapi.loadImgs(meta_image_id)[0]
    file_name = image_info['file_name']
    image_id = int(file_name.split('_')[0])
    region_tlx = int(file_name.split('_')[1].split('x')[0])
    region_tly = int(file_name.split('_')[1].split('x')[1])
    # enlarge_ratio = int(file_name.split('_')[2]) / int(file_name.split('_')[3])
    enlarge_ratio = 4068 / 512
    meta_category_id = meta_result['category_id']
    meta_score = meta_result['score']
    meta_bbox = meta_result['bbox']
    enlarged_bbox = [x * enlarge_ratio for x in meta_bbox]
    category_id = {1: 3, 2: 1, 3: 2, 4: 4}[meta_category_id]
    final_results.append({
        'image_id': image_id,
        'category_id': category_id,
        'bbox_left': int(enlarged_bbox[0] + region_tlx),
        'bbox_top': int(enlarged_bbox[1] + region_tly),
        'bbox_width': int(enlarged_bbox[2]),
        'bbox_height': int(enlarged_bbox[3]),
        'score': meta_score,
    })

with open('results.json', 'w') as fp:
    json.dump(final_results, fp)

