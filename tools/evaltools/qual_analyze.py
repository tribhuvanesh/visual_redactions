#!/usr/bin/python
"""Produce Qualitative comparisons between GT and Predictions.

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
from collections import defaultdict as dd

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread

from pycocotools import mask as mask_utils

from privacy_filters.tools.common.utils import load_attributes, get_image_id_info_index
from privacy_filters.tools.evaltools.evaluate import VISPRSegEval
from privacy_filters.tools.common.image_utils import bimask_to_rgba

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def visualize_img(gt_list, pred_list, out_path):
    """
    For K attributes in the image (pred+GT), creates:
    1. a grid of size (K+1) x 2 where:
        - first row
            - 1st col = image
        - for the next K rows:
            - 1st col = GT mask
            - 2nd col = Pred mask
    :param gt_list: List of GTs
    :param pred_list: List of Predictions
    :return:
    """
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", type=str, help="GT File")
    parser.add_argument("pred_file", type=str, help="Predicted file")
    parser.add_argument("out_dir", type=str, help="Output directory to place output files")
    args = parser.parse_args()
    params = vars(args)

    # Load necessary files ---------------------------------------------------------------------------------------------
    vispr = VISPRSegEval(params['gt_file'], params['pred_file'])
    vispr.evaluate()
    vispr.accumulate()
    vispr.summarize()

    image_id_index = get_image_id_info_index()
    attr_id_to_name, attr_id_to_idx = load_attributes()

    attr_ids = vispr.params.attrIds
    image_ids = vispr.params.imgIds
    n_imgs = len(image_ids)

    out_dir = params['out_dir']
    if not osp.exists(out_dir):
        print 'Directory {} does not exist. Creating it...'.format(out_dir)
        os.mkdir(out_dir)

    # Setup colors
    np.random.seed(42)
    colors = [(np.random.random(size=3) * 255).astype(int) for i in range(40)]
    np.random.shuffle(image_ids)

    # Visualize --------------------------------------------------------------------------------------------------------
    for img_idx, image_id in enumerate(image_ids[:50]):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (img_idx, n_imgs,
                                                                (img_idx * 100.0) / n_imgs))
        sys.stdout.flush()

        out_path = osp.join(out_dir, image_id + '.jpg')

        # For how many attributes do we need to print image? ----------------------------
        # Union of predicted + GT attributes in this image
        this_image_attr = set()
        for attr_id in attr_ids:
            key = (image_id, attr_id)
            if len(vispr._gts[key]) > 0 or len(vispr._pds[key]) > 0:
                this_image_attr.add(attr_id)

        this_image_attr = sorted(list(this_image_attr))
        n_attr = len(this_image_attr)
        nrows = n_attr + 1
        ncols = 2

        fig_width = ncols * 6
        fig_height = nrows * 4

        fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(fig_width, fig_height))

        im = Image.open(image_id_index[image_id]['image_path'])
        w, h = im.size

        # Disable axis everywhere ---------------------------
        for i in range(axarr.shape[0]):
            for j in range(axarr.shape[1]):
                axarr[i, j].axis('off')

        # First row = image ---------------------------------
        ax = axarr[0, 0]
        ax.imshow(im)

        for _idx, attr_id in enumerate(this_image_attr):
            row_idx = _idx + 1
            key = (image_id, attr_id)

            # Plot GT  --------------------------------------
            col_idx = 0
            ax = axarr[row_idx, col_idx]
            if len(vispr._gts[key]) > 0:
                ax.imshow(im, alpha=0.5)
                if _idx == 0:
                    ax.set_title('---- GT ----\n{}'.format(attr_id_to_name[attr_id]))
                else:
                    ax.set_title('{}'.format(attr_id_to_name[attr_id]))
                for inst_idx, det in enumerate(sorted(vispr._gts[key], key=lambda x: -x['area'])):
                    rle = det['segmentation']
                    bimask = mask_utils.decode(rle)
                    inst_mask = bimask_to_rgba(bimask, colors[inst_idx])
                    ax.imshow(inst_mask, alpha=0.8)
                    del bimask
                    del inst_mask

            # Plot Pred --------------------------------------
            col_idx = 1
            ax = axarr[row_idx, col_idx]
            if len(vispr._pds[key]) > 0:
                ax.imshow(im, alpha=0.5)
                if _idx == 0:
                    ax.set_title('---- Pred ----\n{}'.format(attr_id_to_name[attr_id]))
                else:
                    ax.set_title('{}'.format(attr_id_to_name[attr_id]))
                for inst_idx, det in enumerate(sorted(vispr._pds[key], key=lambda x: -x['area'])):
                    rle = det['segmentation']
                    bimask = mask_utils.decode(rle)
                    inst_mask = bimask_to_rgba(bimask, colors[inst_idx])
                    ax.imshow(inst_mask, alpha=0.8)
                    del bimask
                    del inst_mask

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()