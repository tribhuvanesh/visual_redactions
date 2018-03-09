#!/usr/bin/python
"""This is a short description.

Replace this with a more detailed description of what this file contains.
"""
import json
import time
import pickle
import sys
import csv
import argparse
import os
import os.path as osp
import shutil

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread


import _init_paths
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = '/BS/orekondy2/work/opt/FCIS/fcis'
update_config('/BS/orekondy2/work/opt/FCIS/experiments/fcis/cfgs/fcis_coco_demo_tribhu.yaml')
sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))

import scipy
import mxnet as mx
print "use mxnet at", mx.__file__
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting

from pycocotools.coco import COCO
import pycocotools.mask as mask


__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=lambda s: unicode(s, 'utf8'), help="Directory containing list of images")
    parser.add_argument("outfile", type=lambda s: unicode(s, 'utf8'), help="Path to write predictions")
    parser.add_argument("-d", "--device", type=int, default=0, help="Device ID to use")
    args = parser.parse_args()
    params = vars(args)

    # ---------------------------------------------------------- Read config
    ctx_id = [int(i) for i in config.gpus.split(',')]
    pprint.pprint(config)
    sym_instance = eval(config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 81
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    config['gpus'] = str(params['device'])

    # ---------------------------------------------------------- Load Images
    image_path_list = []
    data = []
    scale_factor = 1.0
    img_dir = osp.abspath(params['indir'])
    det_thresh = 0.7

    # Load abs paths of images
    for f in sorted(os.listdir(img_dir)):
        _, f_ext = osp.splitext(f)
        if f_ext in ['.jpg', '.png', '.jpeg']:
            f_path = osp.join(img_dir, f)
            image_path_list.append(f_path)

    print 'Loading {} images into memory...'.format(len(image_path_list))

    for image_path in image_path_list:
        im = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        height, width = im.shape[:2]
        im = cv2.resize(im, (int(scale_factor * width), int(scale_factor * height)))
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    print 'Loaded {} images'.format(len(image_path_list))

    # ---------------------------------------------------------- Predict
    predictions = []

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param('/BS/orekondy2/work/opt/FCIS/model/fcis_coco', 0, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(ctx_id[0])], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # warm up
    for i in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        _, _, _, _ = im_detect(predictor, data_batch, data_names, scales, config)

    # test
    for idx, image_path in enumerate(image_path_list):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, masks, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

        if not config.TEST.USE_MASK_MERGE:
            all_boxes = [[] for _ in xrange(num_classes)]
            all_masks = [[] for _ in xrange(num_classes)]
            nms = py_nms_wrapper(config.TEST.NMS)
            for j in range(1, num_classes):
                indexes = np.where(scores[0][:, j] > 0.7)[0]
                cls_scores = scores[0][indexes, j, np.newaxis]
                cls_masks = masks[0][indexes, 1, :, :]
                try:
                    if config.CLASS_AGNOSTIC:
                        cls_boxes = boxes[0][indexes, :]
                    else:
                        raise Exception()
                except:
                    cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j] = cls_dets[keep, :]
                all_masks[j] = cls_masks[keep, :]
            dets = [all_boxes[j] for j in range(1, num_classes)]
            masks = [all_masks[j] for j in range(1, num_classes)]
        else:
            masks = masks[0][:, 1:, :, :]
            im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
            im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
            print (im_height, im_width)
            boxes = clip_boxes(boxes[0], (im_height, im_width))
            result_masks, result_dets = gpu_mask_voting(masks, boxes, scores[0], num_classes,
                                                        100, im_width, im_height,
                                                        config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                        config.BINARY_THRESH, ctx_id[0])

            dets = [result_dets[j] for j in range(1, num_classes)]
            masks = [result_masks[j][:, 0, :, :] for j in range(1, num_classes)]
        print '{} testing {} {:.4f}s'.format(idx, image_path, toc())
        # visualize
        for i in xrange(len(dets)):
            keep = np.where(dets[i][:, -1] > det_thresh)
            dets[i] = dets[i][keep]
            masks[i] = masks[i][keep]
        im = cv2.imread(image_path_list[idx])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        org_height, org_width = cv2.imread(image_path_list[idx]).shape[:2]
        # im = cv2.resize(im,(int(scale_factor*org_width), int(scale_factor*org_height)))


        """
        visualize all detections in one image
        :param im_array: [b=1 c h w] in rgb
        :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        :param class_names: list of names in imdb
        :param scale: visualize the scaled image
        :return:
        """
        detections = dets
        class_names = classes
        cfg = config
        scale = 1.0

        person_idx = class_names.index('person')
        dets = detections[person_idx]
        msks = masks[person_idx]

        for mask_idx, (det, msk) in enumerate(zip(dets, msks)):
            inst_arr = np.zeros_like(im[:, :, 0])  # Create a 2D W x H array
            bbox = det[:4] * scale
            cod = bbox.astype(int)
            if im[cod[1]:cod[3], cod[0]:cod[2], 0].size > 0:
                msk = cv2.resize(msk, im[cod[1]:cod[3] + 1, cod[0]:cod[2] + 1, 0].T.shape)
                bimsk = (msk >= cfg.BINARY_THRESH).astype('uint8')

                # ------- Create bit-mask for this instance
                inst_arr[cod[1]:cod[3] + 1, cod[0]:cod[2] + 1] = bimsk  # Add thresholded binary mask
                rs_inst_arr = scipy.misc.imresize(inst_arr, (org_height, org_width))
                rle = mask.encode(np.asfortranarray(rs_inst_arr))

                predictions.append({
                    'image_path': image_path,
                    'label': 'person',
                    'segmentation': rle,
                    'bbox': bbox.tolist(),
                    'score': det[-1],
                })

                del msk
                del bimsk
                del rs_inst_arr

    print 'Created {} predictions'.format(len(predictions))

    # ---------------------------------------------------------- Write output
    with open(params['outfile'], 'wb') as wf:
        json.dump(predictions, wf, indent=2)


if __name__ == '__main__':
    main()
