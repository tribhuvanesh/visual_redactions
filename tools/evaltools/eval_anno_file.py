#!/usr/bin/python
"""Evaluates annotation w.r.t a ground-truth.

Given two annotation files (produced from VIA), evaluates them. Optionally, visualizes errors.
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

from privacy_filters.tools.common.utils import get_image_filename_index, clean_via_annotations
from evaltools import  get_mask, via_regions_to_poygons, compute_eval_metrics, visualize_errors

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", type=str, help="Path to GT list of VIA annotations")
    parser.add_argument("pred_file", type=str, help="Path to predicted list of VIA annotations")
    parser.add_argument("-v", "--visualize", type=str, default=None, help="Place visualizations in this directory")
    args = parser.parse_args()

    params = vars(args)
    # print 'Input parameters: '
    # print json.dumps(params, indent=2)

    img_filename_index = get_image_filename_index()
    gt_via = clean_via_annotations(params['gt_file'], img_fname_index=img_filename_index)
    pred_via = clean_via_annotations(params['pred_file'], img_fname_index=img_filename_index)

    gt_via_fname_set = set([e['filename'] for k, e in gt_via.iteritems()])
    pred_via_fname_set = set([e['filename'] for k, e in pred_via.iteritems()])

    if gt_via_fname_set != pred_via_fname_set:
        print 'GT contains #annotations: ', len(gt_via_fname_set)
        print 'Pred contains #annotations: ', len(pred_via_fname_set)
        print '# common annotations: ', len(gt_via_fname_set & pred_via_fname_set)
        print 'Computing metrics over common annotations ...'

    common_fname_set = gt_via_fname_set & pred_via_fname_set

    num_skipped = 0

    precision_list = []
    recall_list = []
    iou_list = []

    for key in common_fname_set:
        gt_anno = gt_via[key]
        pred_anno = pred_via[key]

        this_filename = gt_anno['filename']

        gt_regions = gt_anno['regions']
        pred_regions = pred_anno['regions']

        # Evaluate only if both files contains at least one region
        if len(gt_regions) == 0 or len(pred_regions) == 0:
            num_skipped += 1
            continue

        gt_polygons = via_regions_to_poygons(gt_regions)
        pred_polygons = via_regions_to_poygons(pred_regions)

        img_path = gt_anno['filepath']
        im = Image.open(img_path)
        w, h = im.size

        gt_mask = get_mask(w, h, gt_polygons)
        pred_mask = get_mask(w, h, pred_polygons)

        this_precision, this_recall, this_iou = compute_eval_metrics(gt_mask, pred_mask)

        precision_list.append(this_precision)
        recall_list.append(this_recall)
        iou_list.append(this_iou)

        if params['visualize'] is not None:
            vis_out_dir = params['visualize']
            if not osp.exists(vis_out_dir):
                print 'Path {} does not exist. Creating it...'.format(vis_out_dir)
                os.mkdir(vis_out_dir)

            img_out_path = osp.join(vis_out_dir, this_filename)
            metrics_text = 'Precision = {:.2f}   '.format(100 * this_precision)
            metrics_text += 'Recall    = {:.2f}   '.format(100 * this_recall)
            metrics_text += 'IoU       = {:.2f}\n\n'.format(100 * this_iou)
            visualize_errors(im, gt_mask, pred_mask, img_out_path, metrics_text)


    print
    print 'Evaluating over {} images: '.format(len(precision_list))
    print 'Mean Precision = {:.2f}'.format(100 * np.mean(precision_list))
    print 'Mean Recall    = {:.2f}'.format(100 * np.mean(recall_list))
    print 'Mean IoU       = {:.2f}'.format(100 * np.mean(iou_list))


if __name__ == '__main__':
    main()