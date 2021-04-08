# Author: t-zhong
# Create: 2021/4/3

import json
import os
import xml.etree.ElementTree as ET
from glob import glob
from shutil import copyfile

from tqdm import tqdm

val_size = 0.2

base_dir = os.path.expanduser('~/datasets/fruit-images-for-object-detection')
train_annos_path = os.path.join(base_dir, 'train_zip/train/*.xml')
test_annos_path = os.path.join(base_dir, 'test_zip/test/*.xml')
train_annos_list = glob(train_annos_path)
test_annos_list = glob(test_annos_path)

os.makedirs('datasets/fruit/annotations', exist_ok=True)
os.makedirs('datasets/fruit/train', exist_ok=True)
os.makedirs('datasets/fruit/val', exist_ok=True)
os.makedirs('datasets/fruit/test', exist_ok=True)

image_id = -1
bndbox_id = -1
categories_dict = {'apple': 1, 'banana': 2, 'orange': 3}
categories = [{'supercategory': 'none', 'id': 1, 'name': 'apple'}, {'supercategory': 'none', 'id': 2, 'name': 'banana'}, {'supercategory': 'none', 'id': 3, 'name': 'orange'}]
train_images = []
val_images = []
train_annotations = []
val_annotations = []

for idx, train_anno_file in tqdm(enumerate(train_annos_list)):
    split = 'val' if (idx + 1) % int(1 / val_size) == 0 else 'train'
    train_anno = ET.parse(train_anno_file)
    root = train_anno.getroot()
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    image_id += 1
    image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
    image_path = train_anno_file.replace('.xml', '.jpg')
    if split == 'train':
        copyfile(image_path, os.path.join('datasets/fruit/train', os.path.basename(image_path)))
        train_images.append(image)
    else:
        copyfile(image_path, os.path.join('datasets/fruit/val', os.path.basename(image_path)))
        val_images.append(image)
    objects = root.findall('object')

    for obj in objects:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bndbox_id += 1
        anno = {'area': 0, 'iscrowd': 0, 'image_id': image_id, 'bbox': [xmin, ymin, xmax-xmin, ymax-ymin], 'category_id': categories_dict[name], 'id': bndbox_id, 'ignore': 0, 'segmentation': []}
        if split == 'train':
            train_annotations.append(anno)
        else:
            val_annotations.append(anno)

test_images = []

for idx, test_anno_file in tqdm(enumerate(test_annos_list)):
    train_anno = ET.parse(test_anno_file)
    root = train_anno.getroot()
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    image_id += 1
    image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
    image_path = test_anno_file.replace('.xml', '.jpg')
    test_images.append(image)
    copyfile(image_path, os.path.join('datasets/fruit/test', os.path.basename(image_path)))


with open('datasets/fruit/annotations/instances_train.json', 'w') as fp:
    json.dump({'images': train_images, 'annotations': train_annotations, 'categories': categories}, fp)

with open('datasets/fruit/annotations/instances_val.json', 'w') as fp:
    json.dump({'images': val_images, 'annotations': val_annotations, 'categories': categories}, fp)

with open('datasets/fruit/annotations/image_info_test.json', 'w') as fp:
    json.dump({'images': test_images, 'categories': categories}, fp)
