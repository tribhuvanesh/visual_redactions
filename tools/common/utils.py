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
# Override above if file contains regions
override_if_regions_exist = True


def load_attributes(v1_attributes=False):
    """
    Return mappings of:
    a) attribute_id -> attribute_name
    b) attribute_id (a string eg., a4_gender) -> attribute_idx (a number)
    :return:
    """
    if v1_attributes:
        attributes_path = osp.join(DS_ROOT, 'attributes.tsv')
    else:
        attributes_path = osp.join(SEG_ROOT, 'attributes.tsv')
    attr_id_to_name = dict()
    attr_id_to_idx = dict()

    with open(attributes_path, 'r') as fin:
        ts = csv.DictReader(fin, delimiter='\t')
        rows = filter(lambda r: r['idx'] is not '', [row for row in ts])

        for row in rows:
            attr_id_to_name[row['attribute_id']] = row['description']
            attr_id_to_idx[row['attribute_id']] = int(row['idx'])

    return attr_id_to_name, attr_id_to_idx


def labels_to_vec(labels, attr_id_to_idx):
    n_labels = len(attr_id_to_idx)
    label_vec = np.zeros(n_labels)
    for attr_id in labels:
        label_vec[attr_id_to_idx[attr_id]] = 1
    return label_vec


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
    Example pre-cleaned annotation: https://pastebin.com/cP3RCS3i
    Example post-cleaned annotation file: https://pastebin.com/8ifs3RxM
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
            ignore_file = True
            # This annotation entry contains one of the ignore attributes
            n_regions = 0
            if override_if_regions_exist:
                n_regions += len(entry['regions'])
                if 'full_scan' in entry['file_attributes']:
                    n_regions += 1
                if n_regions > 0:
                    ignore_file = False
            if not ignore_file:
                pass
            elif not return_all:
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
            full_region_dct["shape_attributes"]["all_points_x"] = [scan_margin, w - scan_margin, w - scan_margin,
                                                                   scan_margin,
                                                                   scan_margin]
            full_region_dct["shape_attributes"]["all_points_y"] = [scan_margin, scan_margin, h - scan_margin,
                                                                   h - scan_margin,
                                                                   scan_margin]
            # Add this region to the list of existing regions
            if len(entry['regions'].keys()) > 0:
                next_region_id = max(map(int, entry['regions'].keys())) + 1
            else:
                next_region_id = 0
            entry['regions'][str(next_region_id)] = full_region_dct

        shapes_in_anno = set([region_dct['shape_attributes']['name'] for k, region_dct in entry['regions'].iteritems()])

        # This is of the format: {u'cy': 484, u'cx': 1078, u'r': 38, u'name': u'circle'}
        if 'circle' in shapes_in_anno:
            for k, region_dct in entry['regions'].iteritems():
                if region_dct['shape_attributes']['name'] == 'circle':
                    new_region_dct = {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [],
                            "all_points_y": [],
                            "all_ts": []
                        },
                        "region_attributes": region_dct['region_attributes']
                    }
                    _cx, _cy = region_dct['shape_attributes']['cx'], region_dct['shape_attributes']['cy']
                    _r = region_dct['shape_attributes']['r']

                    # No. of polygons to represent this circle
                    n_circle_vertices = 16
                    for i in range(n_circle_vertices):
                        angle = (2.0 * np.pi / n_circle_vertices) * i
                        _x = _cx + _r * np.cos(angle)
                        _y = _cy + _r * np.sin(angle)
                        new_region_dct['shape_attributes']['all_points_x'].append(_x)
                        new_region_dct['shape_attributes']['all_points_y'].append(_y)
                    # Close the loop
                    new_region_dct['shape_attributes']['all_points_x'].append(
                        new_region_dct['shape_attributes']['all_points_x'][0])
                    new_region_dct['shape_attributes']['all_points_y'].append(
                        new_region_dct['shape_attributes']['all_points_y'][0])

                    entry['regions'][k] = new_region_dct

        # This is of the format: {u'y': 273, u'x': 300, u'height': 13, u'name': u'rect', u'width': 77}
        if 'rect' in shapes_in_anno:
            for k, region_dct in entry['regions'].iteritems():
                if region_dct['shape_attributes']['name'] == 'rect':
                    new_region_dct = {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [],
                            "all_points_y": [],
                            "all_ts": []
                        },
                        "region_attributes": region_dct['region_attributes']
                    }
                    _x, _y = region_dct['shape_attributes']['x'], region_dct['shape_attributes']['y']
                    _h, _w = region_dct['shape_attributes']['height'], region_dct['shape_attributes']['width']

                    new_region_dct['shape_attributes']['all_points_x'] = [_x, _x+_w, _x+_w, _x, _x]
                    new_region_dct['shape_attributes']['all_points_y'] = [_y, _y, _y+_h, _y+_h, _y]

                    entry['regions'][k] = new_region_dct

        # Add an instance id to each region
        # Each region either: (a) contains an instance_id attribute OR (b) does not contain it
        # In case of (b), we need to assign it a random id

        # Generate some random IDs
        rand_ids = range(500)
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

    # Remove spurious polygons i.e., with just 2 points
    for key, entry in via_cleaned_anno.iteritems():
        for k, region_dct in entry['regions'].iteritems():
            if region_dct['shape_attributes']['name'] == 'polygon':
                if len(region_dct['shape_attributes']['all_points_x']) < 3:
                    del via_cleaned_anno[key]['regions'][k]
                    break

    # Sometimes VIA uses negative values for points close to the border. Replace them with 0s
    for key, entry in via_cleaned_anno.iteritems():
        for k, region_dct in entry['regions'].iteritems():
            if region_dct['shape_attributes']['name'] == 'polygon':
                all_x = region_dct['shape_attributes']['all_points_x']
                all_y = region_dct['shape_attributes']['all_points_y']
                if any([z < 0. for z in all_x + all_y]):
                    all_pos_x = [max(0., x) for x in all_x]
                    all_pos_y = [max(0., y) for y in all_y]
                    region_dct['shape_attributes']['all_points_x'] = all_pos_x
                    region_dct['shape_attributes']['all_points_y'] = all_pos_y

    # Remove file attributes with blank values
    for key, entry in via_cleaned_anno.iteritems():
        emtpy_k_list = []
        for k, v in entry['file_attributes'].iteritems():
            if v == '':
                emtpy_k_list.append(k)
        for k in emtpy_k_list:
            del via_cleaned_anno[key]['file_attributes'][k]

        # FIX: Annotator used '[6-10]' and '[10+]' as file labels. Replace them
        if '[6-10]' in entry['file_attributes'].keys():
            entry['file_attributes']['crowd_6-10'] = entry['file_attributes'].pop('[6-10]')
        if '[10+]' in entry['file_attributes'].keys():
            entry['file_attributes']['crowd_10+'] = entry['file_attributes'].pop('[10+]')
        if '[Unsure]' in entry['file_attributes'].keys():
            entry['file_attributes']['unsure'] = entry['file_attributes'].pop('[Unsure]')

    return via_cleaned_anno
