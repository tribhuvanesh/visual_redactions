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

from sklearn.metrics import auc

from privacy_filters import SEG_ROOT
from privacy_filters.tools.common.utils import load_attributes_shorthand
from privacy_filters.tools.evaltools.evaluate_simple import IGNORE_ATTR, MODE_TO_ATTR_ID

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def correct_precision_recall(precision, recall, eval_at_zero_recall=False):
    """
    Fix Pascal VOC 2010 style - Make precision decrease monotonically
    Sec 3.4.1: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html

    :param precision: List of precision scores
    :param recall: List of recall scores
    :param eval_at_zero_recall: Use maximum precision at 0 recall
    :return:
    """
    precision = np.asarray(precision)
    recall = np.asarray(recall)

    # Add 0s
    if eval_at_zero_recall:
        precision = np.append(precision, 0.0)
        recall = np.append(recall, 0.0)

    # Sort by recall
    sort_idxs = np.argsort(recall)
    recall = recall[sort_idxs]
    precision = precision[sort_idxs]

    for i in range(len(precision)):
        r = recall[i]  # r
        # Indices where r' >= r
        rp_idxs = np.where(recall >= r)
        if len(rp_idxs) > 0:
            precision[i] = np.max(precision[rp_idxs])

    return precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_file", type=str, help="Path to precision recall scores")
    args = parser.parse_args()
    params = vars(args)

    attr_id_to_name = load_attributes_shorthand()

    results = json.load(open(params['eval_file']))

    for _mode, attr_ids in MODE_TO_ATTR_ID.iteritems():
        attr_names = []
        aps = []
        for attr_id in attr_ids:
            precision = results['per_attribute'][attr_id]['precision']
            recall = results['per_attribute'][attr_id]['recall']

            if not (isinstance(precision, list) and isinstance(recall, list)):
                precision = [precision, ]
                recall = [recall, ]

            precision, recall = correct_precision_recall(precision, recall, eval_at_zero_recall=True)
            ap = auc(recall, precision) * 100.0

            attr_names.append(attr_id_to_name[attr_id])
            aps.append(ap)

        print '--- Mode = ', _mode, '---'
        print '\t'.join(attr_names)
        print '\t'.join(map(str, aps))
        print 'mAP = ', np.mean(aps)
        print


if __name__ == '__main__':
    main()