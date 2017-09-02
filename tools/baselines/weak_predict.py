#!/usr/bin/python
"""Predict weakly-supervised localization cues.

Given a GAP-like model, obtain weakly-supervised signals and write masks.
What this script does:
    for each image:
        for each new_attribute (idx > 67):
            append mask to existing masks
        for each (predicted/gt) attribute (above some threshold):
            get bimask for this attribute (41x41)
            resize bimask
            obtain instances from bimask
            append instances to list of predictions

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
from scipy import ndimage
from scipy.misc import imread, imresize

from pycocotools import mask

from privacy_filters.tools.common.image_utils import get_image_size
from privacy_filters.tools.common.utils import load_attributes
from privacy_filters.tools.common.adapter_utils import prev_to_new_attr_vec, prev_to_new_masks
from privacy_filters.tools.common.utils import load_attributes, get_image_id_info_index, labels_to_vec

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

ATTR_THRESH = 0.01   # Predict attribute only if it is above this prob. threshold
CAM_THRESH = 0.2   # Same as used in the CAM paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="List of images for which to produce masks")
    parser.add_argument("pkl_file", type=str, help="Pickled file containing {image_id -> (localization_masks, "
                                                   "attr_probs)} information")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    parser.add_argument("-u", "--use_attributes", type=str, default=None, help="Use only these attributes")
    parser.add_argument("-g", "--gt_labels", action='store_true', default=False,
                        help="Create GT-label based predictions instead (i.e., ignore attribute predictions from "
                             "CAM PAP model)")
    parser.add_argument("-i", "--instances", action='store_true', default=False, help="Predict instances in each image")
    args = parser.parse_args()

    params = vars(args)

    # image_id -> {image_path, anno_path, fold}
    image_id_index = get_image_id_info_index()
    _, attr_id_to_idx_v1 = load_attributes(v1_attributes=True)
    idx_to_attr_id_v1 = {v: k for k, v in attr_id_to_idx_v1.iteritems()}
    _, attr_id_to_idx = load_attributes()
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}

    if params['use_attributes'] is None:
        attr_set_use = set(attr_id_to_idx.keys())
    else:
        attr_set_use = set(map(lambda x: x.strip(), open(params['use_attributes']).readlines()))

    # Load image_ids ---------------------------------------------------------------------------------------------------
    image_id_set = set()
    with open(params['infile']) as f:
        for _line in f:
            _, image_name = osp.split(_line.strip())
            image_id, ext = osp.splitext(image_name)
            image_id_set.add(image_id)

            # Load GT attributes too
            anno = json.load(open(image_id_index[image_id]['anno_path']))
            gt_vec_v1 = labels_to_vec(anno['labels'], attr_id_to_idx_v1)
            gt_vec = prev_to_new_attr_vec(gt_vec_v1, attr_id_to_idx_v1, attr_id_to_idx)
            image_id_index[image_id]['gt_vec'] = gt_vec

    # Load weakly supervised predictions -------------------------------------------------------------------------------
    print 'Loading weakly supervised masks...'
    # {image_id -> (localization_masks, attr_probs)}
    #               where localization_masks = 68x41x41 matrix
    #                     attr_probs = 68 vector
    image_id_to_info = pickle.load(open(params['pkl_file']))
    predictions = []

    print 'Creating predictions from masks...'
    # Weakly supervised CAM masks -> instance predictions --------------------------------------------------------------
    n_files = len(image_id_set)
    for idx, image_id in enumerate(image_id_set):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (idx, n_files,
                                                                (idx * 100.0) / n_files))
        sys.stdout.flush()
        loc_masks_v1, pred_probs_v1 = image_id_to_info[image_id]
        pred_probs = prev_to_new_attr_vec(pred_probs_v1, attr_id_to_idx_v1, attr_id_to_idx)
        gt_labels = image_id_index[image_id]['gt_vec']

        w, h = get_image_size(image_id_index[image_id]['image_path'])

        # loc_masks is a 68x41x41 mask. Infer and append masks for 6 new attributes
        loc_masks = prev_to_new_masks(loc_masks_v1, attr_id_to_idx_v1, attr_id_to_idx)

        for this_attr_idx, this_attr_prob in enumerate(pred_probs):
            this_attr_id = idx_to_attr_id[this_attr_idx]
            prob_score = gt_labels[this_attr_idx] if params['gt_labels'] else this_attr_prob
            if (this_attr_id not in attr_set_use) or (prob_score < ATTR_THRESH):
                continue

            lowres_mask = loc_masks[this_attr_idx]   # 41 x 41 mask
            # Binarize this mask
            lowres_bimask = (lowres_mask > CAM_THRESH * np.max(lowres_mask)).astype('uint8')
            lowres_prediction_list = []
            if params['instances']:
                # Predict attribute for each contiguous blob (c) using connected components
                labeled_mask, nr_objects = ndimage.label(lowres_bimask > 0)
                for inst_id in range(1, nr_objects):  # inst_id = 0 indicates background
                    instance_mask = (labeled_mask == inst_id).astype('uint8')
                    lowres_prediction_list.append(instance_mask)
            else:
                lowres_prediction_list.append(lowres_bimask)
            del lowres_mask
            for lowres_inst_mask in lowres_prediction_list:
                # Resize mask
                highres_inst_mask = imresize(lowres_inst_mask, [h, w], interp='bilinear', mode='F')
                highres_inst_bimask = np.asfortranarray((highres_inst_mask > (CAM_THRESH * np.max(highres_inst_mask)))
                                                        .astype('uint8'))
                rle = mask.encode(highres_inst_bimask)
                predictions.append({
                    'image_id': image_id,
                    'attr_id': this_attr_id,
                    'segmentation': rle,
                    'score': prob_score,
                })
                del highres_inst_mask
                del highres_inst_bimask
            del lowres_prediction_list

    # Dump predictions -------------------------------------------------------------------------------------------------
    print 'Writing {} predictions to file: {}'.format(len(predictions), params['outfile'])
    with open(params['outfile'], 'wb') as wf:
        json.dump(predictions, wf, indent=2)


if __name__ == '__main__':
    main()