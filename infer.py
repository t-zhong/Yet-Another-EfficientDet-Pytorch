# Modified from ./efficientdet_test.py ./coco_eval.py
# Autoor t-zhong

"""
Simple script predicts bounding box for given image(s), store the bounding box in json file, 
and drawing bounding box if needed.
"""
import argparse
import os
import time
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from matplotlib import colors
from torch.backends import cudnn

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import boolean_string, preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def display(preds, imgs, imshow=True, imwrite=False, **kwargs):
    """
    Draw bounding box on the given image(s). If there are bounding boxes on given image(s),
    display and write the given image(s) to disk if needed.
    """
    if kwargs.get('image_paths'):
        image_paths = kwargs['image_paths']
    else:
        image_paths = [str(i) for i in range(0, len(imgs))]
        
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            print('no object has been detected in {image_path}')
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int32)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(
                f'results/{opt.project}/efficientdet_d{opt.compound_coef}/{os.path.basename(image_paths[i])}', imgs[i])


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
parser.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
parser.add_argument('-s', '--source', type=str, default=None, help='image(s) to be infered, should be an image path or a directory')
parser.add_argument('--force_input_size', type=int, default=None, help='force use original input size if setted')
# parser.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
parser.add_argument('--cuda', type=boolean_string, default=True)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--float16', type=boolean_string, default=False)
parser.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')

opt = parser.parse_args()

project = opt.project
compound_coef = opt.compound_coef
force_input_size = opt.force_input_size

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

params = Params(f'projects/{opt.project}.yml')
anchor_ratios = eval(params.anchors_ratios)
anchor_scales = eval(params.anchors_scales)
obj_list = params.obj_list

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=anchor_ratios, scales=anchor_scales)
if opt.weights is None:
    weights_path = sorted(glob(f'logs/{project}/efficientdet-d{compound_coef}*'), key=os.path.getmtime)[-6]
else:
    weights_path = opt.weigths
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

if opt.source is None:
    opt.source = f'datasets/{project}/val'
if os.path.isdir(opt.source):
    image_paths = glob(os.path.join(opt.source, '*.jpg'))
else:
    image_paths = list(opt.source)

os.makedirs(f'results/{opt.project}/efficientdet_d{compound_coef}', exist_ok=True)

results = []

for image_path in image_paths:
    ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        preds = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
        preds = invert_affine(framed_metas, preds)
    
    scores = preds['scores']
    class_ids = preds['class_ids']
    rois = preds['rois']

    if rois.shape[0] > 0:
        # x1,y1,x2,y2 -> x1,y1,w,h
        rois[:, 2] -= rois[:, 0]
        rois[:, 3] -= rois[:, 1]

        bbox_score = scores

        for roi_id in range(rois.shape[0]):
            score = float(bbox_score[roi_id])
            label = int(class_ids[roi_id])
            box = rois[roi_id, :]

            image_result = {
                'image_id': image_id,
                'category_id': label + 1,
                'score': float(score),
                'bbox': box.tolist(),
            }

            results.append(image_result)
    
    display(preds, ori_imgs, imshow=False, imwrite=True, image_paths=[image_path])


##### no speed test required.
# print('running speed test...')
# with torch.no_grad():
#     print('test1: model inferring and postprocessing')
#     print('inferring image for 10 times...')
#     t1 = time.time()
#     for _ in range(10):
#         _, regression, classification, anchors = model(x)

#         out = postprocess(x,
#                           anchors, regression, classification,
#                           regressBoxes, clipBoxes,
#                           threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)

#     t2 = time.time()
#     tact_time = (t2 - t1) / 10
#     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

# uncomment this if you want a extreme fps test
# print('test2: model inferring only')
# print('inferring images for batch_size 32 for 10 times...')
# t1 = time.time()
# x = torch.cat([x] * 32, 0)
# for _ in range(10):
#     _, regression, classification, anchors = model(x)
#
# t2 = time.time()
# tact_time = (t2 - t1) / 10
# print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
