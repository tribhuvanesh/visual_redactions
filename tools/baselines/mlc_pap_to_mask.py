#!/usr/bin/python
"""Returns segmentation maps from PAP model[1].

Given a trained Privacy Attribute Prediction (PAP) model[1], predicts instance segmentation masks for it.
Here, we simply provide a mask spanning the entire image if the attribute is predicted.

[1] "Towards a Visual Privacy Advisor: Understanding and Predicting Privacy Risks in Images."
Orekondy, Tribhuvanesh, Bernt Schiele, and Mario Fritz
ICCV 2017
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

# https://github.com/BVLC/caffe/issues/438
from skimage import io

from pycocotools import mask

from privacy_filters.tools.common.adapter_utils import prev_to_new_attr_vec
from privacy_filters.tools.common.utils import load_attributes, get_image_filename_index, labels_to_vec
from privacy_filters.tools.common.image_utils import get_image_size

io.use_plugin('matplotlib')

from privacy_filters import DS_ROOT, SEG_ROOT, CAFFE_ROOT

sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe/layers')  # the datalayers we will use are in this directory.
sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe')  # the tools file is in this folder

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

THRESH = 0.01


def dct_to_mask_list(filename_to_probs, fname_index, idx_to_attr_id, attr_set_use):
    prediction_list = []
    for filename, probs in filename_to_probs.iteritems():
        image_path = fname_index[filename]
        file_id, ext = osp.splitext(filename)
        w, h = get_image_size(image_path)
        bimask = np.ones((h, w), order='F', dtype='uint8')
        rle = mask.encode(bimask)
        del bimask
        for this_attr_idx, this_attr_prob in enumerate(probs):
            this_attr_id = idx_to_attr_id[this_attr_idx]
            if this_attr_id in attr_set_use and this_attr_prob > THRESH:
                score_dct = {
                    'image_id': file_id,
                    'attr_id': this_attr_id,
                    'segmentation': rle,
                    'score': this_attr_prob,
                }
                prediction_list.append(score_dct)
    return prediction_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="List of images for which to produce masks")
    parser.add_argument("pap_file", type=str, help="PAP file")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    parser.add_argument("-u", "--use_attributes", type=str, default=None, help="Use only these attributes")
    parser.add_argument("-g", "--gt_preds", type=str, default=None, help="Write masks predicted using GT too")
    args = parser.parse_args()

    params = vars(args)

    # Load image_ids ---------------------------------------------------------------------------------------------------
    image_id_set = set()
    with open(params['infile']) as f:
        for _line in f:
            _, image_name = osp.split(_line.strip())
            image_id, ext = osp.splitext(image_name)
            image_id_set.add(image_id)

    filename_to_probs = {}
    filename_to_gt = {}

    # Load PAP results -------------------------------------------------------------------------------------------------
    _, attr_id_to_idx_v1 = load_attributes(v1_attributes=True)
    idx_to_attr_id_v1 = {v: k for k, v in attr_id_to_idx_v1.iteritems()}
    _, attr_id_to_idx = load_attributes()
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}

    with open(params['pap_file']) as f:
        for line in f:
            jf = json.loads(line.strip())
            anno_file = jf['anno_path']
            anno = json.load(open(anno_file))
            image_id = anno['id']
            _, filename = osp.split(anno['image_path'])
            if image_id in image_id_set:
                pred_probs = np.asarray(jf['pred_probs'])
                gt_labels = anno['labels']
                gt_vec = labels_to_vec(gt_labels, attr_id_to_idx_v1)

                # Convert to new format and write
                filename_to_probs[filename] = prev_to_new_attr_vec(pred_probs, attr_id_to_idx_v1, attr_id_to_idx)
                filename_to_gt[filename] = prev_to_new_attr_vec(gt_vec, attr_id_to_idx_v1, attr_id_to_idx)

    # Predict masks from attributes ------------------------------------------------------------------------------------
    # Create a mask spanning the entire image for each predicted attribute
    # Required format in: privacy_filters/tools/scripts/evaluate.py
    if params['use_attributes'] is None:
        attr_set_use = set(attr_id_to_idx.keys())
    else:
        attr_set_use = set(map(lambda x: x.strip(), open(params['use_attributes']).readlines()))

    n_files = len(filename_to_probs)
    n_attr = len(attr_set_use)
    fname_index = get_image_filename_index()

    # Write PAP Predicted mask -----------------------------------------------------------------------------------------
    print 'Writing masks for {} attributes and {} files'.format(n_attr, n_files)
    prediction_list = dct_to_mask_list(filename_to_probs, fname_index, idx_to_attr_id, attr_set_use)

    with open(params['outfile'], 'w') as wf:
        json.dump(prediction_list, wf, indent=2)

    # Optionally, write GT Predicted mask ------------------------------------------------------------------------------
    if params['gt_preds'] is not None:
        print 'Writing masks for {} attributes and {} files'.format(n_attr, n_files)
        prediction_list = dct_to_mask_list(filename_to_gt, fname_index, idx_to_attr_id, attr_set_use)

        with open(params['gt_preds'], 'w') as wf:
            json.dump(prediction_list, wf, indent=2)


if __name__ == '__main__':
    main()
