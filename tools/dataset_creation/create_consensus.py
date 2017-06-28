#!/usr/bin/python
"""Create a consensus dataset.

Create a set of images, sampling N images per attribute.
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

N_PER_ATTR = 10    # No. of images per attribute


def main():
    # --- Setup paths --------------------------------------------------------------------------------------------------
    # Images here are organized per attribute
    images_dir = osp.join(SEG_ROOT, 'images', 'all')
    # Where to place sampled images
    out_dir = osp.join(SEG_ROOT, 'consensus', 'images')
    # Mapping of old and new path
    img_map = osp.join(SEG_ROOT, 'consensus', 'mapping.tsv')

    # Read all attribute IDs
    attr_id_list = []
    with open(osp.join(SEG_ROOT, 'attributes.tsv')) as f:
        f.readline()   # Skip header
        for line in f:
            items = line.strip().split()
            this_attr_id = items[1]
            attr_id_list.append(this_attr_id)

    np.random.seed(42)

    # --- Copy images --------------------------------------------------------------------------------------------------
    with open(img_map, 'w') as mapf:
        for attr_id in attr_id_list:
            print attr_id
            attr_batch_img_dir = osp.join(images_dir, attr_id)
            # Images here are stored in batches
            # So: make a list of all images accessing each batch
            all_fnames = []   # Store a list of: (batch_id, filename)
            for b_id in os.listdir(attr_batch_img_dir):
                batch_files = os.listdir(osp.join(attr_batch_img_dir, b_id))
                all_fnames += [(b_id, fname) for fname in batch_files]
            idx = np.random.choice(len(all_fnames), size=N_PER_ATTR, replace=False)
            sel_fnames = [all_fnames[i] for i in idx]

            for b_id, fname in sel_fnames:
                # Choose new name
                img_path = osp.join(images_dir, attr_id, b_id, fname)
                new_fname = '{}_{}'.format(attr_id, fname)
                new_fname = fname

                # Copy image
                shutil.copy(img_path, new_img_path)

                # Write mapping
                mapf.write('{}\t{}\t{}\n'.format(attr_id, img_path, new_img_path))


if __name__ == '__main__':
    main()