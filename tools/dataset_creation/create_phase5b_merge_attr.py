#!/usr/bin/python
"""Merge attributes, pool images from them and batch them into a new directory.

Given
  a) fold
  b) two attributes to merge
pools images from SEG_ROOT/phase4/images/<fold>/<attr*> and writes to a new merged directory
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

from privacy_filters import DS_ROOT, SEG_ROOT
from privacy_filters.tools.common.utils import get_image_filename_index, clean_via_annotations, load_attributes

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

BATCH_SIZE = 100


def phase4_merge_attr(fold_name, attr_ids_to_merge, new_attr_id):
    # --- Setup paths --------------------------------------------------------------------------------------------------
    # Filename -> Filepath
    img_filename_index = get_image_filename_index()
    # Load attributes
    attr_id_to_name, attr_id_to_idx = load_attributes()
    # Source phase4 images
    anno_attr_out_dir = osp.join(SEG_ROOT, 'phase5b', 'images', fold_name)
    # Output directory
    attr_out_dir = osp.join(anno_attr_out_dir, new_attr_id)

    # --- Gather images ------------------------------------------------------------------------------------------------
    full_image_set = set()
    for attr_id in attr_ids_to_merge:
        attr_in_dir = osp.join(anno_attr_out_dir, attr_id)
        if not osp.exists(attr_in_dir):
            # raise ValueError('Attribute {} does not exist'.format(attr_in_dir))
            print('WARNING: Attribute {} does not exist'.format(attr_in_dir))
        for root, dirs, files in os.walk(attr_in_dir):
            for image_name in files:
                this_img_path = img_filename_index[image_name]
                full_image_set.add(this_img_path)

    # --- Get existing images ------------------------------------------------------------------------------------------
    existing_image_set = set()
    if not osp.exists(attr_out_dir):
        # If this is the first time the script is being run
        print '{} does not exist. Creating it...'.format(attr_out_dir)
        os.makedirs(osp.join(attr_out_dir, '0'))
    else:
        # Walk and get a list of existing images
        for root, dirs, files in os.walk(attr_out_dir):
            for image_name in files:
                this_img_path = img_filename_index[image_name]
                existing_image_set.add(this_img_path)

    print 'Found {} existing images'.format(len(existing_image_set))
    print '# Duplicate images = {}'.format(len(full_image_set & existing_image_set))
    print '# Images to write = {}'.format(len(full_image_set - existing_image_set))
    out_image_set = full_image_set - existing_image_set

    # --- Write images -------------------------------------------------------------------------------------------------
    n_written = 0
    for img_path in out_image_set:
        # Which batch to write in?
        last_batch_id = max(map(int, os.listdir(attr_out_dir)))
        if len(os.listdir(osp.join(attr_out_dir, str(last_batch_id)))) >= BATCH_SIZE:
            # If full, create the next directory
            last_batch_id += 1
            os.mkdir(osp.join(attr_out_dir, str(last_batch_id)))

        this_img_attr_out_dir = osp.join(attr_out_dir, str(last_batch_id))
        _, img_filename = osp.split(img_path)
        img_dst = osp.join(this_img_attr_out_dir, img_filename)
        os.symlink(img_path, img_dst)
        n_written += 1

    print '# Written = ', n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold_name", type=str, help="train/val/test", choices=['val2017', 'train2017', 'test2017'])
    parser.add_argument("new_attr_name", type=str, help="New attribute ID")
    parser.add_argument("prev_attr", type=str, help="Attribute IDs to merge", nargs='+')
    args = parser.parse_args()

    params = vars(args)
    phase4_merge_attr(params['fold_name'], params['prev_attr'], params['new_attr_name'])


if __name__ == '__main__':
    main()