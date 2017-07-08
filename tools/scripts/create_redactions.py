#!/usr/bin/python
"""Create redacted versions of annotated images.

Given an annotation file, produces the following versions, with each instance in the image:
  a. outlined
  b. blurred
  c. blackened
  d. image of the cropped region + gray-scale filter
  e. a square crop of the image + gray-scale filter
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

from privacy_filters.tools.common.image_utils import draw_outline_on_img, fill_region
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
    if not osp.exists(out_dir):
        print 'Path {} does not exist. Creating it...'.format(out_dir)
        os.makedirs(out_dir)

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

        file_id, file_ext = osp.splitext(this_filename)

        polygons, instances = via_regions_to_polygons(regions, include_instance=True)

        # Iterate over instances
        for instance_idx in set(instances):
            this_polygon_list = []
            # Instances may be over multiple polygons. What are the polygons belonging to this instance?
            for _poly, _inst_idx in zip(polygons, instances):
                if _inst_idx == instance_idx:
                    this_polygon_list.append(_poly)

            # A. Outline
            outline_img = im.copy()
            for poly in this_polygon_list:
                outline_img = draw_outline_on_img(outline_img, poly, color='yellow', width=4)
            out_filename = '{}-{}-outline{}'.format(file_id, instance_idx, file_ext)
            out_path = osp.join(out_dir, out_filename)
            outline_img.save(out_path)

            # B. Blurred
            pass

            # C. Redacted region
            redacted_img = im.copy()
            for poly in this_polygon_list:
                redacted_img = fill_region(redacted_img, poly, color='yellow')
            out_filename = '{}-{}-redacted{}'.format(file_id, instance_idx, file_ext)
            out_path = osp.join(out_dir, out_filename)
            redacted_img.save(out_path)


if __name__ == '__main__':
    main()