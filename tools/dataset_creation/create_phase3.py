#!/usr/bin/python
"""Create an attribute consensus directory, with around 10-20 images per attribute

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

N_SAMPL = 20


def create_phase4_fold(fold_name):
    # --- Setup paths --------------------------------------------------------------------------------------------------
    # Where to place batched images
    out_dir = osp.join(SEG_ROOT, 'phase3', 'images', fold_name)
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

    # Shortlist only few images per attribute
    for attr_id in attr_id_to_img:
        if len(attr_id_to_img[attr_id]) > N_SAMPL:
            attr_id_to_img[attr_id] = np.random.choice(attr_id_to_img[attr_id], size=N_SAMPL, replace=False)

    # --- Copy images --------------------------------------------------------------------------------------------------
    for attr_id in attr_id_to_img:
        attr_out_dir = osp.join(out_dir, attr_id)
        if not osp.exists(attr_out_dir):
            print 'Directory {} does not exist. Creating directory ...'.format(attr_out_dir)
            os.makedirs(attr_out_dir)

        for img_path in attr_id_to_img[attr_id]:
            _, filename = osp.split(img_path)
            dst_path = osp.join(attr_out_dir, filename)
            os.symlink(img_path, dst_path)


def main():
    np.random.seed(42)
    for fold in ['val2017', 'train2017', 'test2017']:
        print '** Processing ', fold
        create_phase4_fold(fold)


if __name__ == '__main__':
    main()
