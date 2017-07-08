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
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread

from privacy_filters import DS_ROOT, SEG_ROOT
from privacy_filters.tools.common.image_utils import get_image_size

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

# Ignore annotation if it contains any of these file-level attributes
IGNORE_IF_FILE_ATTR = {
    'discard',
    'unsure',
    'crowd_6-10',
    'crowd_10+'
}


def get_image_filename_index():
    """
    Obtain a mapping of filename -> filepath for images
    :return:
    """
    index_path = osp.join(SEG_ROOT, 'privacy_filters', 'cache', 'fname_index.pkl')
    if osp.exists(index_path):
        print 'Found cached index. Loading it...'
        return pickle.load(open(index_path, 'rb'))
    else:
        print 'Creating filename index ...'
        fname_index = dict()
        images_dir = osp.join(DS_ROOT, 'images')
        for fold in os.listdir(images_dir):
            for img_filename in os.listdir(osp.join(images_dir, fold)):
                image_path = osp.join(images_dir, fold, img_filename)
                fname_index[img_filename] = image_path
        pickle.dump(fname_index, open(index_path, 'wb'))
        return fname_index


def clean_via_annotations(anno_path, img_fname_index=None, return_all=True):
    """
    Clean and add some additional info to via annotations.
    Example annotation: https://pastebin.com/cP3RCS3i
    :param anno_path:
    :param img_fname_index:
    :param return_all: If true, returns all entries of the annotation file. Otherwise, drops certain entries based on
    file-level attributes (e.g., crowds or discards)
    :return:
    """
    if img_fname_index is None:
        img_fname_index = get_image_filename_index()

    with open(anno_path) as jf:
        via_anno = json.load(jf)

    via_cleaned_anno = dict()

    # The annotations are indexed by <filename><file_size>. Fix this to just <filename>.
    # Additionally, add filepath to entry
    for key, entry in via_anno.iteritems():
        this_img_filename = entry['filename']

        this_file_level_attr = set(entry['file_attributes'].keys())
        if len(this_file_level_attr & IGNORE_IF_FILE_ATTR) > 0:
            # This annotation entry contains one of the ignore attributes
            if not return_all:
                # Simply ignore this entry
                continue

        # I sometimes add attr_id to beginning of filename for readability. Strip it if exists
        # For example: a0_safe_2017_15285423.jpg
        if not this_img_filename.startswith('2017'):
            prefix = this_img_filename.split('2017')[0]
            this_img_filename = this_img_filename.replace(prefix, '')
            entry['filename'] = this_img_filename

        if 'filepath' not in entry:
            this_img_filepath = img_fname_index[this_img_filename]
            entry['filepath'] = this_img_filepath

        via_cleaned_anno[this_img_filename] = entry

        # Convert file-level attribute 'full_scan' to a polygon spanning entire region
        if 'full_scan' in entry['file_attributes']:
            # What's the image dimension?
            w, h = get_image_size(entry['filepath'])
            # Use this margin (in pixels)
            scan_margin = 1
            # Construct region
            full_region_dct = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [],
                    "all_points_y": [],
                    "all_ts": []
                },
                "region_attributes": {}
            }
            full_region_dct["shape_attributes"]["all_points_x"] = [scan_margin, w-scan_margin, w-scan_margin, scan_margin,
                                                                   scan_margin]
            full_region_dct["shape_attributes"]["all_points_y"] = [scan_margin, scan_margin, h-scan_margin, h-scan_margin,
                                                                   scan_margin]
            # Add this region to the list of existing regions
            next_region_id = max(map(int, entry['regions'].keys())) + 1
            entry['regions'][str(next_region_id)] = full_region_dct

        # TODO Convert Bounding boxes to polygon

        # Add an instance id to each region
        # Each region either: (a) contains an instance_id attribute OR (b) does not contain it
        # In case of (b), we need to assign it a random id

        # Generate some random IDs
        rand_ids = range(100)
        # Mapping to tag instance "p_N" to an instance id
        tag_to_instance_id = dict()
        for k, region_dct in entry['regions'].iteritems():
            if 'instance_id' in region_dct['region_attributes']:
                # This region has been tagged with a "p_N"
                if region_dct['region_attributes']['instance_id'] in tag_to_instance_id:
                    this_instance_id = tag_to_instance_id[region_dct['region_attributes']['instance_id']]
                else:
                    this_instance_id = rand_ids.pop(0)
                    tag_to_instance_id[region_dct['region_attributes']['instance_id']] = this_instance_id
            else:
                this_instance_id = rand_ids.pop(0)
            region_dct['assigned_instance_id'] = this_instance_id

    return via_cleaned_anno
