#!/usr/bin/python
"""This is a short description.

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

import datetime
from collections import defaultdict as dd

from privacy_filters.tools.common.anno_utils import AnnoEncoder

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def anno_stats(file_id_to_img_anno):
    """
    Computes some statistics over image annotations
    :param file_id_to_img_anno:
    :return:
    """
    stats_dct = dict()
    # 1. Number of images
    stats_dct['n_images'] = len(file_id_to_img_anno)
    # 2. attr -> #images
    # 3. attr -> #instances
    attr_id_to_n_img = dd(int)
    attr_id_to_n_inst = dd(int)
    for file_id, anno_entry in file_id_to_img_anno.iteritems():
        for attr_entry in anno_entry['attributes']:
            attr_id_to_n_inst[attr_entry['attr_id']] += 1
        file_attr = set([attr_entry['attr_id'] for attr_entry in anno_entry['attributes']])
        for attr_id in file_attr:
            attr_id_to_n_img[attr_id] += 1
    stats_dct['attr_id_to_n_img'] = attr_id_to_n_img
    stats_dct['attr_id_to_n_inst'] = attr_id_to_n_inst
    stats_dct['n_attr'] = len(attr_id_to_n_inst)
    stats_dct['present_attr'] = sorted(attr_id_to_n_inst.keys())
    return stats_dct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outname", type=str, help="Path to merged annotations file")
    parser.add_argument("folds", type=str, nargs='+', help="folds")
    args = parser.parse_args()

    assert not osp.exists(args.outname)

    merged_file_id_to_anno = dict()

    for fold_path in args.folds:
        this_anno_dct = json.load(open(fold_path))['annotations']
        print '# Annotations in {} = {}'.format(fold_path, len(this_anno_dct))
        merged_file_id_to_anno = dict(merged_file_id_to_anno.items() + this_anno_dct.items())

    print '# Annotations after merge = {}'.format(len(merged_file_id_to_anno))
    print 'Writing to ', args.outname
    anno_to_write = {'annotations': merged_file_id_to_anno,
                     'created_at': str(datetime.datetime.now()),
                     'stats': anno_stats(merged_file_id_to_anno),
                     'merged_from': args.folds}
    with open(args.outname, 'wb') as wjf:
        json.dump(anno_to_write, wjf, indent=2, cls=AnnoEncoder)


if __name__ == '__main__':
    main()
