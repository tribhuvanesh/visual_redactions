#!/usr/bin/python
"""Create directories to annotate the privacy attributes.

Given:
 i. Phase 2 Person annotations (+ crowd labels, etc.)
 ii. Multi-label annotations
create image directories for attribute segmentation annotation.
In the process, we skip images which contains crowd labels (since it's expensive to annotate).
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
from privacy_filters.tools.common.utils import get_image_filename_index, clean_via_annotations, load_attributes

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

BATCH_SIZE = 100


def create_phase4_fold(fold_name):
    # --- Setup paths --------------------------------------------------------------------------------------------------
    # Location of annotated batches
    anno_batch_dir = osp.join(SEG_ROOT, 'phase2', 'annotations', fold_name)
    # Annotation list
    mlc_anno_list_path = osp.join(DS_ROOT, fold_name + '.txt')
    # Filename -> Filepath
    img_filename_index = get_image_filename_index()
    # Load attributes
    attr_id_to_name, attr_id_to_idx = load_attributes()
    # Where to place the phase4 images
    anno_attr_out_dir = osp.join(SEG_ROOT, 'phase4', 'images', fold_name)

    # --- Create a mapping of attr_id <-> [list of images, ...] --------------------------------------------------------
    attr_id_to_img = dd(list)
    img_to_attr_id = dd(list)

    print 'Creating attr_id->[img_path, ] mapping ...'
    with open(mlc_anno_list_path) as f:
        for line_idx, _line in enumerate(f):
            anno_path = osp.join(DS_ROOT, _line.strip())
            with open(anno_path) as jf:
                anno = json.load(jf)
                image_path = osp.join(DS_ROOT, anno['image_path'])
                for attr_id in anno['labels']:
                    attr_id_to_img[attr_id].append(image_path)
                    img_to_attr_id[image_path].append(attr_id)

    # --- Scan anno_batch_dir and detect which images are already present ----------------------------------------------
    # This prevents copying duplicates
    attr_id_to_existing_img = dd(set)
    for attr_id in attr_id_to_idx.keys():
        for root, dirs, files in os.walk(osp.join(anno_attr_out_dir, attr_id)):
            for image_name in files:
                this_img_path = img_filename_index[image_name]
                # Sanity check. Not image should be repeated for the same attribute
                assert(this_img_path not in attr_id_to_existing_img[attr_id])
                attr_id_to_existing_img[attr_id].add(this_img_path)
        if len(attr_id_to_existing_img[attr_id]) == 0:
            attr_id_to_existing_img[attr_id] = set()

    # --- Iterate through VIA annotations and place them accordingly ---------------------------------------------------
    n_written = 0
    n_skipped = 0
    n_dupl = 0
    batch_anno_filenames = os.listdir(anno_batch_dir)
    for batch_fname in sorted(batch_anno_filenames, key=lambda x: int(osp.splitext(x)[0])):
        # Iterate over each batch
        batch_filepath = osp.join(anno_batch_dir, batch_fname)
        via_list = clean_via_annotations(batch_filepath, img_fname_index=img_filename_index, return_all=True)
        via_fname_set = set([e['filename'] for k, e in via_list.iteritems()])

        for file_id, entry in via_list.iteritems():
            img_path = entry['filepath']
            file_attr_dct = entry['file_attributes']

            # Skip this image if: a) it contains crowd attributes b) contains an unsure tag
            skip_image = False
            skip_file_attr = {'crowd_6-10', 'crowd_10+', 'unsure'}
            if len(set(file_attr_dct.keys()) & skip_file_attr) > 0:
                skip_image = True

            if skip_image:
                n_skipped += 1
                continue

            # For each attribute present in this image, symlink the image to the respective phase4 directory
            img_attrs = img_to_attr_id[img_path]
            for attr_id in img_attrs:
                if img_path in attr_id_to_existing_img[attr_id]:
                    # Skip image if it's already present/annotated in phase 4
                    n_dupl += 1
                    continue

                attr_out_dir = osp.join(anno_attr_out_dir, attr_id)
                if not osp.exists(attr_out_dir):
                    print '{} does not exist. Creating it...'.format(attr_out_dir)
                    os.mkdir(attr_out_dir)
                #  ---- Which batch to place it in?
                if len(os.listdir(attr_out_dir)) == 0:
                    # If this is the first time this is being run, create the first directory
                    os.mkdir(osp.join(attr_out_dir, '0'))
                # Get the last batch_id
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
    print '# Skipped (crowds/unsure) = ', n_skipped
    print '# Duplicates skipped = ', n_dupl


def main():
    # for fold in ['val2017', 'train2017', 'test2017']:
    for fold in ['test2017']:
        print '** Processing ', fold
        create_phase4_fold(fold)


if __name__ == '__main__':
    main()