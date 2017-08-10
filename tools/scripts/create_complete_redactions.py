#!/usr/bin/python
"""Create redacted versions of annotated images.

Given an annotation file, produces the following versions, for ALL instances in the image:
  a. outlined
  b. blurred
  c. blackened
  d. original
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

from PIL import Image, ImageDraw, ImageOps
from scipy.misc import imread

from privacy_filters.tools.common.image_utils import draw_outline_on_img, fill_region, blur_region, crop_region, \
    rgba_to_rgb
from privacy_filters.tools.common.utils import get_image_filename_index, clean_via_annotations
from privacy_filters.tools.evaltools.evaltools import via_regions_to_polygons

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
    via_list = clean_via_annotations(params['anno_file'], img_fname_index=img_filename_index, return_all=False)
    via_fname_set = set([e['filename'] for k, e in via_list.iteritems()])

    num_skipped = 0

    out_dir = params['out_dir']
    for strat in ['outline', 'blurred', 'redacted', 'original']:
        out_subdir = osp.join(out_dir, strat)
        if not osp.exists(out_subdir):
            print 'Path {} does not exist. Creating it...'.format(out_subdir)
            os.makedirs(out_subdir)

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
        inst_w, int_h = im.size

        polygons, instances = via_regions_to_polygons(regions, include_instance=True)

        # a. Outline
        outline_img = im.copy()
        for poly in polygons:
            outline_img = draw_outline_on_img(outline_img, poly, color='yellow', width=4)
        out_path = osp.join(out_dir, 'outline', this_filename)
        outline_img.save(out_path)

        # b. Blurred
        blurred_img = im.copy()
        for poly in polygons:
            blurred_img = blur_region(blurred_img, poly, radius=10)
        out_path = osp.join(out_dir, 'blurred', this_filename)
        blurred_img.save(out_path)

        # c. Redacted
        redacted_img = im.copy()
        for poly in polygons:
            redacted_img = fill_region(redacted_img, poly, color='black')
        out_path = osp.join(out_dir, 'redacted', this_filename)
        redacted_img.save(out_path)

        # d. Original
        out_path = osp.join(out_dir, 'original', this_filename)
        im.save(out_path)


if __name__ == '__main__':
    main()
