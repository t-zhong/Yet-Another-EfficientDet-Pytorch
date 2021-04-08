# Modified from ./coco_eval.py and ./efficientdet_test.py
# Autoor t-zhong

"""
Simple script predicts bounding box for given image(s), store the bounding box in json file, 
and drawing bounding box if needed.
"""

import argparse
import json
import os
from glob import glob

import cv2
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string, plot_one_box, get_index_label, color_list

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def detect(images_dir, targets_dir, image_ids, cocoapi, model, threshold=0.2):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = cocoapi.loadImgs(image_id)[0]
        image_path = images_dir + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

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
        ori_imgs = ori_imgs.copy()
        for j in range(len(preds['rois'])):
            (x1, y1, x2, y2) = preds['rois'][j].astype(int)
            obj = obj_list[preds['class_ids'][j]]
            score = float(preds['scores'][j])

            plot_one_box(ori_imgs[0], [x1, y1, x2, y2], label=obj, score=score, color=color_list[get_index_label(obj, obj_list)])
        cv2.imwrite(f'{targets_dir}/{image_info["file_name"]}', ori_imgs[0])

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'results/{project_name}/{os.path.basename(weights_path)}/results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
    ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
    ap.add_argument('--cuda', type=boolean_string, default=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--float16', type=boolean_string, default=False)
    ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
    args = ap.parse_args()
    print(args)

    compound_coef = args.compound_coef
    nms_threshold = args.nms_threshold
    use_cuda = args.cuda
    gpu = args.device
    use_float16 = args.float16
    override_prev_results = args.override
    project_name = args.project

    if args.weights is None:
        weights_path = sorted(glob(f'logs/{project_name}/efficientdet-d{compound_coef}*'), key=os.path.getmtime)[-1]
    else:
        weights_path = args.weights

    params = yaml.safe_load(open(f'projects/{project_name}.yml'))
    obj_list = params['obj_list']
    image_info_file = f'datasets/{params["project_name"]}/annotations/image_info_test.json'
    images_dir = f'datasets/{params["project_name"]}/test/'
    image_info = COCO(image_info_file)
    image_ids = image_info.getImgIds()
    # if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                    ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(gpu)

        if use_float16:
            model.half()
    
    targets_dir = f'results/{project_name}/{os.path.basename(weights_path)}'
    os.makedirs(targets_dir, exist_ok=True)

    detect(images_dir, targets_dir, image_ids, image_info, model)
