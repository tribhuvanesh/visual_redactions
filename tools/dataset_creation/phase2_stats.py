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


def get_phase2_stats(fold_name, csv_out_path):
    # --- Setup paths --------------------------------------------------------------------------------------------------
    # Location of annotated batches
    anno_batch_dir = osp.join(SEG_ROOT, 'phase2', 'annotations', fold_name)
    # Annotation list
    mlc_anno_list_path = osp.join(DS_ROOT, fold_name + '.txt')
    # Filename -> Filepath
    img_filename_index = get_image_filename_index()
    # Load attributes
    attr_id_to_name, attr_id_to_idx = load_attributes()

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

    # --- Create a mapping of image_path -> {crowd_6-10, crowd_10+, person, None} --------------------------------------
    # Also, perform some sanity checks
    img_to_label_type = dict()
    batch_anno_filenames = os.listdir(anno_batch_dir)
    for batch_fname in sorted(batch_anno_filenames, key=lambda x: int(osp.splitext(x)[0])):
        # Iterate over each batch
        batch_filepath = osp.join(anno_batch_dir, batch_fname)
        try:
            via_list = clean_via_annotations(batch_filepath, img_fname_index=img_filename_index, return_all=True)
        except ValueError:
            continue
        via_fname_set = set([e['filename'] for k, e in via_list.iteritems()])

        for file_id, entry in via_list.iteritems():
            img_path = entry['filepath']
            file_attr_dct = entry['file_attributes']
            if len(entry['regions']) > 0:
                img_to_label_type[img_path] = 'person'

                if len(file_attr_dct) > 0:
                    print 'Warning: {} contains regions and tags ({})'.format(file_id, file_attr_dct)

            elif len(file_attr_dct) > 0:
                # Sanity check
                if len(file_attr_dct) > 1:
                    print 'Warning: {} contains multiple file attributes ({})'.format(file_id, file_attr_dct)

                # if 'crowd_6-10' in file_attr_dct.keys() or 'crowd_10+' in file_attr_dct.keys():
                img_to_label_type[img_path] = file_attr_dct.keys()[0]

            else:
                img_to_label_type[img_path] = 'none'

    # --- Write stats --------------------------------------------------------------------------------------------------
    # A) label_type -> # images
    print
    tot_images = len(img_to_label_type)
    print '# Total images = ', tot_images
    for label_type in sorted(set(img_to_label_type.values())):
        this_count = img_to_label_type.values().count(label_type)
        print '# {} = \t{} ({:.2f} %)'.format(label_type, this_count, (this_count*100.0)/tot_images)

    # B) attr -> # images
    attr_stats_dct = dict()
    fieldnames = set()
    anno_img_set = set(img_to_label_type.keys())
    for attr_id, attr_img_list in attr_id_to_img.iteritems():
        attr_img_set = set(attr_img_list)

        this_attr_stats = dd(int)

        this_attr_stats['attr_id'] = attr_id
        this_attr_stats['attr_name'] = attr_id_to_name[attr_id]
        this_attr_stats['n_vispr'] = len(attr_img_list)
        this_attr_stats['n_common'] = len(anno_img_set.intersection(attr_img_set))

        for img in (anno_img_set.intersection(attr_img_set)):
            this_attr_stats[img_to_label_type[img]] += 1

        attr_stats_dct[attr_id] = this_attr_stats
        for k in this_attr_stats.keys():
            fieldnames.add(k)

    with open(csv_out_path, 'w') as wf:
        writer = csv.DictWriter(wf, fieldnames=sorted(list(fieldnames)))

        attr_list = sorted(attr_stats_dct.keys(), key=lambda x: int(x.split('_')[0][1:]))

        writer.writeheader()
        for k in attr_list:
            writer.writerow(attr_stats_dct[k])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=str, help="fold name", choices=['val2017', 'train2017', 'test2017'])
    parser.add_argument("out_csv", type=str, default='/dev/null', help="Place visualizations in this directory")
    args = parser.parse_args()

    get_phase2_stats(args.fold, args.out_csv)


if __name__ == '__main__':
    main()
