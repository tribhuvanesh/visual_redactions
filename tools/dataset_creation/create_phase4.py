#!/usr/bin/python
"""Create an attribute consensus directory, with symlink-ing respective images per attribute

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

from privacy_filters import DS_ROOT, SEG_ROOT

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

BATCH_SIZE = 100


def create_phase4_fold(fold_name):
    # --- Setup paths --------------------------------------------------------------------------------------------------
    # Where to place batched images
    out_dir = osp.join(SEG_ROOT, 'phase2', 'images', fold_name)
    # Annotation list
    mlc_anno_list_path = osp.join(DS_ROOT, fold_name + '.txt')

    # --- Create a mapping of attr_id -> [list of images, ...] ---------------------------------------------------------
    attr_id_to_img = dd(list)

    print 'Creating attr_id->[img_path, ] mapping ...'
    with open(mlc_anno_list_path) as f:
        for line_idx, _line in enumerate(f):
            anno_path = osp.join(DS_ROOT, _line.strip())
            with open(anno_path) as jf:
                anno = json.load(jf)
                image_path = osp.join(DS_ROOT, anno['image_path'])
                for attr_id in anno['labels']:
                    attr_id_to_img[attr_id].append(image_path)

    # --- Copy images --------------------------------------------------------------------------------------------------
    for attr_id in attr_id_to_img:
        print 'Processing: ', attr_id
        num_images = len(attr_id_to_img[attr_id])
        num_batches = (num_images / BATCH_SIZE) + 1
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_images)

            batch_out_dir = osp.join(out_dir, attr_id, batch_idx)

            for img_path in attr_id_to_img[attr_id][start_idx:end_idx]:
                _, filename = osp.split(img_path)
                dst_path = osp.join(batch_out_dir, filename)
                os.symlink(img_path, dst_path)


def main():
    for fold in ['val2017', 'train2017', 'test2017']:
        print '** Processing ', fold
        create_phase4_fold(fold)
