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

from multiprocessing import Pool

from privacy_filters.tools.evaltools.evaluate_simple import VISPRSegEvalSimple, MODE_TO_ATTR_ID

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", type=str, help="GT File")
    parser.add_argument("pred_file", type=str, help="Predicted file")
    parser.add_argument("-w", "--write_results", action='store_true', default=True,
                        help="Write summarized results into JSON in predicted file directory")
    parser.add_argument("-n", "--nthresh", type=int, default=5,
                        help="Number of thresholds")
    args = parser.parse_args()
    params = vars(args)
    print
    print 'Evaluating: ', params['pred_file']

    # --- Decide thresholds --------------------------------------------------------------------------------------------
    # A. Get a list of scores
    preds = json.load(open(params['pred_file']))
    scores = [x['score'] for x in preds]

    # B. Set thresholds
    nthresh = params['nthresh']
    # Option 1: Uniformly sample in this range
    thresh_vals = np.linspace(min(scores), max(scores) - 1e-3, num=nthresh)
    print 'Computing over over range: ', thresh_vals

    # --- Calculate scores over thresholds -----------------------------------------------------------------------------
    tau_and_vispr = []
    for tau in thresh_vals:
        print '----- Calculating results @ tau = {:.3f} -----'.format(tau)
        vispr = VISPRSegEvalSimple(params['gt_file'], params['pred_file'])
        vispr.params.score_thresh = tau
        vispr.evaluate()
        vispr.accumulate()
        vispr.summarize()

        tau_and_vispr.append((tau, vispr))

    idx_to_attr_id = dict()
    for attr_idx, attr_id in enumerate(tau_and_vispr[0][1].params.attrIds):
        idx_to_attr_id[attr_idx] = attr_id
    attr_id_to_idx = {v: k for k, v in idx_to_attr_id.iteritems()}
    attr_ids = attr_id_to_idx.keys()

    # --- Collate multiple scores --------------------------------------------------------------------------------------
    out_dct = dict()
    thesh_to_vispr = dict(tau_and_vispr)   # Sort by thresh val
    out_dct['thresholds'] = thresh_vals.tolist()
    # --- Overall stats
    out_dct['overall'] = {
        'precision': [np.mean(thesh_to_vispr[tau].overall_stats['precision']) for tau in thresh_vals],
        'recall': [np.mean(thesh_to_vispr[tau].overall_stats['recall']) for tau in thresh_vals],
        'iou': [np.mean(thesh_to_vispr[tau].overall_stats['iou']) for tau in thresh_vals],
    }

    # --- Per Mode
    out_dct['per_mode'] = dict()
    for _mode in MODE_TO_ATTR_ID.keys():
        out_dct['per_mode'][_mode] = dict()
        for metric in ['precision', 'recall', 'iou']:
            out_dct['per_mode'][_mode][metric] = [np.mean([thesh_to_vispr[tau].overall_stats[metric][attr_id_to_idx[attr_id]]
                                                          for attr_id in MODE_TO_ATTR_ID[_mode]]) for tau in thresh_vals]

    # --- Per Attribute
    out_dct['per_attribute'] = dict()
    for attr_id in attr_ids:
        out_dct['per_attribute'][attr_id] = dict()
        for metric in ['precision', 'recall', 'iou']:
            out_dct['per_attribute'][attr_id][metric] = [thesh_to_vispr[tau].overall_stats[metric][attr_id_to_idx[attr_id]]
                                                         for tau in thresh_vals]

    # --- Write --------------------------------------------------------------------------------------------------------
    pred_path, pred_nameext = osp.split(params['pred_file'])
    pred_name, _ext = osp.splitext(pred_nameext)
    out_name = pred_name + '-curves.json'
    out_path = osp.join(pred_path, out_name)
    json.dump(out_dct, open(out_path, 'w'), indent=2, sort_keys=True)
    print 'Results written to: ', out_path


if __name__ == '__main__':
    main()