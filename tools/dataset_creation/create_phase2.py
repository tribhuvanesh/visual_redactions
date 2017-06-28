#!/usr/bin/python
"""Move non-safe images and batch them for annotating persons.

DS_ROOT/images contains images separated into train, val and test.
 This script maintains:
 a. Maintains this mapping
 b. Batches images
 c. Skips `safe` images
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

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

BATCH_SIZE = 100
SAFE_LABEL = 'a0_safe'


def create_phase2_fold(fold_name):
    # --- Setup paths --------------------------------------------------------------------------------------------------
    # Where to place batched images
    out_dir = osp.join(SEG_ROOT, 'phase2', 'images', fold_name)
    # Annotation list
    mlc_anno_list_path = osp.join(DS_ROOT, fold_name + '.txt')

    # Store the list of images here
    image_path_list = []

    n_total_img = 0
    n_nonsafe_img = 0
    print 'Reading image paths...'
    with open(mlc_anno_list_path) as f:
        for line_idx, _line in enumerate(f):
            n_total_img += 1
            anno_path = osp.join(DS_ROOT, _line.strip())
            with open(anno_path) as jf:
                anno = json.load(jf)
                if SAFE_LABEL not in anno['labels']:
                    img_path = osp.join(DS_ROOT, anno['image_path'])
                    image_path_list.append(img_path)
                    n_nonsafe_img += 1

    print '# Images found = ', n_total_img
    print '# Non-safe images = ', n_nonsafe_img

    n_batches = (n_nonsafe_img / BATCH_SIZE) + 1
    print '# Batches = ', n_batches

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(image_path_list))

        batch_out_dir = osp.join(out_dir, str(batch_idx))
        if not osp.exists(batch_out_dir):
            os.makedirs(batch_out_dir)

        for img_path in image_path_list[start_idx:end_idx]:
            _, img_fname = osp.split(img_path)
            img_dst = osp.join(batch_out_dir, img_fname)
            os.symlink(img_path, img_dst)

        sys.stdout.write("Processing %d/%d (%.2f%% done)   \r" % (batch_idx, n_batches, batch_idx * 100.0 / n_batches))
        sys.stdout.flush()


def main():
    for fold in ['val2017', 'train2017', 'test2017']:
        print 'Processing ', fold
        create_phase2_fold(fold)
        print

if __name__ == '__main__':
    main()