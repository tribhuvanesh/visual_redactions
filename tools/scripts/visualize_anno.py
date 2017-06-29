#!/usr/bin/python
"""Given a VIA annotation file, produce overlays to visualize annotations.

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

from privacy_filters.tools.common.utils import get_image_filename_index, clean_via_annotations
from privacy_filters.tools.common.image_utils import resize_min_side
from privacy_filters.tools.evaltools.evaltools import get_mask, via_regions_to_polygons, compute_eval_metrics, \
    visualize_errors, resize_polygons, visualize_masks, visualize_polygons

from privacy_filters.tools.common. timer import Timer

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anno_file", type=str, help="Path to list of VIA annotations")
    parser.add_argument("out_dir", type=str, default=None, help="Place visualizations in this directory")
    args = parser.parse_args()

    params = vars(args)
    # print 'Input parameters: '
    # print json.dumps(params, indent=2)

    img_filename_index = get_image_filename_index()
    via_list = clean_via_annotations(params['anno_file'], img_fname_index=img_filename_index)
    via_fname_set = set([e['filename'] for k, e in via_list.iteritems()])

    num_skipped = 0

    for key in via_fname_set:
        anno = via_list[key]
        this_filename = anno['filename']
        regions = anno['regions']

        if len(regions) == 0:
            # Visualize only if it contains a region
            num_skipped += 1
            continue

        img_path = anno['filepath']
        im = Image.open(img_path)
        w, h = im.size

        polygons, instances = via_regions_to_polygons(regions, include_instance=True)

        # ------------ Scale image and polygons to a smaller size to reduce computation
        scaled_im = resize_min_side(im, 760)
        scaled_w, scaled_h = scaled_im.size

        x_shrink_factor = scaled_w/float(w)
        y_shrink_factor = scaled_h / float(h)

        polygons = resize_polygons(polygons,
                                      x_shrink_factor=x_shrink_factor,
                                      y_shrink_factor=y_shrink_factor)
        w, h, im = scaled_w, scaled_h, scaled_im

        mask_list = get_mask(w, h, polygons, return_grid_list=True)

        # ------------ Produce visualizations
        vis_out_dir = params['out_dir']
        if not osp.exists(vis_out_dir):
            print 'Path {} does not exist. Creating it...'.format(vis_out_dir)
            os.mkdir(vis_out_dir)

        img_out_path = osp.join(vis_out_dir, this_filename)
        visualize_polygons(im, polygons, img_out_path, instances=instances)


if __name__ == '__main__':
    main()