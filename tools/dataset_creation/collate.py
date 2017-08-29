#!/usr/bin/python
"""Combine VIA annotations and store them in Cityscapes-like format.

Obtain individual annotations from:
  a. Phase 2 (Persons)
  b. Phase 4 (Other attributes)
and collate them and store these annotations in SEG_ROOT/annotations.

(Similar to Object Instance Annotations in MS-COCO: http://mscoco.org/dataset/#download)
Format of file:
    {
        'image_id': '2017_235123.jpg',
        'image_path': 'images/train2017/2017_235123.jpg'
        'image_height': 1024,
        'image_width' : 2048,
        attributes: [
            {
                'id':           4,
                'attr_id':      'a105_face_all',
                'polygons': [[], ],          # polygon [[x1 y1 x2 y2, ...], [x1 y1 x2 y2, ...], ]
                'area':         float,               #
                'bbox':         [x, y, width, height],
                'iscrowd' :     0 or 1,
            }
        ]
    }
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
import re
import datetime
from collections import defaultdict as dd

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread

from privacy_filters import DS_ROOT, SEG_ROOT
from privacy_filters.tools.common.anno_utils import AttributeAnnotation, ImageAnnotation, AnnoEncoder
from privacy_filters.tools.common.image_utils import get_image_size
from privacy_filters.tools.common.utils import get_image_filename_index, clean_via_annotations, load_attributes

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

PERSON_ATTR_ID = 'a109_person_body'


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
        for attr_entry in anno_entry.attributes:
            attr_id_to_n_inst[attr_entry.attr_id] += 1
        file_attr = set([attr_entry.attr_id for attr_entry in anno_entry.attributes])
        for attr_id in file_attr:
            attr_id_to_n_img[attr_id] += 1
    stats_dct['attr_id_to_n_img'] = attr_id_to_n_img
    stats_dct['attr_id_to_n_inst'] = attr_id_to_n_inst
    stats_dct['n_attr'] = len(attr_id_to_n_inst)
    stats_dct['present_attr'] = sorted(attr_id_to_n_inst.keys())
    return stats_dct


def collate(fold_name, snapshot_name):
    # --- Setup paths --------------------------------------------------------------------------------------------------
    # Location of annotated batches - Persons
    phase2_batch_dir = osp.join(SEG_ROOT, 'phase2', 'annotations', fold_name)
    # Location of annotated batches - Other Attributes
    phase4_batch_dir = osp.join(SEG_ROOT, 'phase4', 'annotations', fold_name)
    # Filename -> Filepath
    img_filename_index = get_image_filename_index()
    # Out directory
    final_out_dir = osp.join(SEG_ROOT, 'annotations', snapshot_name)
    final_out_path = osp.join(final_out_dir, '{}.json'.format(fold_name))
    assert not osp.exists(final_out_path), 'Output path {} exists. Delete it and try again.'.format(final_out_path)

    # --- Get Person Annotations ---------------------------------------------------------------------------------------
    # Create a mapping of file_id -> ImageAnnotation
    file_id_to_img_anno = dict()
    n_written = 0
    n_skipped = 0
    n_dupl = 0
    batch_anno_filenames = os.listdir(phase2_batch_dir)
    print 'Processing attribute "Persons"... '
    for batch_idx, batch_fname in enumerate(sorted(batch_anno_filenames, key=lambda x: int(osp.splitext(x)[0]))):
        # Iterate over each batch
        batch_filepath = osp.join(phase2_batch_dir, batch_fname)
        via_list = clean_via_annotations(batch_filepath, img_fname_index=img_filename_index, return_all=True)

        for file_id, entry in via_list.iteritems():
            img_path = entry['filepath']
            w, h = get_image_size(img_path)
            file_attr_dct = entry['file_attributes']

            # Skip this image if: a) it contains crowd attributes b) contains an unsure tag c) does not contain regions
            skip_image = False
            skip_file_attr = {'crowd_6-10', 'crowd_10+', 'unsure'}
            if len(set(file_attr_dct.keys()) & skip_file_attr) > 0:
                skip_image = True

            if len(entry['regions']) < 1:
                skip_image = True

            if skip_image:
                n_skipped += 1
                continue

            # -- At this point, this anno blob *should* contain regions
            ainst_id_to_attr_anno = dict()
            # Iterate over each anno region
            for region in entry['regions'].values():
                all_points_x = region['shape_attributes']['all_points_x']
                all_points_y = region['shape_attributes']['all_points_y']
                assigned_instance_id = region['assigned_instance_id']
                # Squish x and y into [x1 y1 x2 y2 ...]
                polygon = [z for xy_tup in zip(all_points_x, all_points_y) for z in xy_tup]
                if assigned_instance_id in ainst_id_to_attr_anno:
                    ainst_id_to_attr_anno[assigned_instance_id].add_polygon(polygon)
                else:
                    try:
                        this_attr_anno = AttributeAnnotation(assigned_instance_id, PERSON_ATTR_ID, [polygon, ])
                    except AssertionError:
                        print file_id, batch_filepath
                        raise
                    ainst_id_to_attr_anno[assigned_instance_id] = this_attr_anno

            # Create an ImageAnnotation object for this image
            this_img_anno = ImageAnnotation(file_id, img_path, h, w)
            for attr_anno in ainst_id_to_attr_anno.values():
                this_img_anno.add_attribute_annotation(attr_anno)
            assert file_id not in file_id_to_img_anno
            file_id_to_img_anno[file_id] = this_img_anno

    # --- Get Annotations for other Attributes -------------------------------------------------------------------------
    # Walk through the remaining attributes and add these annotations to batch of ImageAnnotations
    attr_list = os.listdir(phase4_batch_dir)
    for attr_id in attr_list:
        attr_batch_dir = osp.join(phase4_batch_dir, attr_id)
        batch_anno_filenames = os.listdir(attr_batch_dir)
        if len(batch_anno_filenames) > 0:
            # Skip batches which were used for consensus (e.g., 0_abc.json)
            batch_anno_filenames = filter(lambda x: re.search('^[0-9]+$', osp.splitext(x)[0]), batch_anno_filenames)
            print 'Processing attribute "{}" (# Batches = {})... '.format(attr_id, len(batch_anno_filenames))
        for batch_idx, batch_fname in enumerate(sorted(batch_anno_filenames, key=lambda x: int(osp.splitext(x)[0]))):
            # Iterate over each batch
            batch_filepath = osp.join(phase4_batch_dir, attr_id, batch_fname)
            via_list = clean_via_annotations(batch_filepath, img_fname_index=img_filename_index, return_all=True)

            for file_id, entry in via_list.iteritems():
                img_path = entry['filepath']
                w, h = get_image_size(img_path)
                file_attr_dct = entry['file_attributes']

                # Skip this image if: a) contains an unsure tag c) does not contain regions
                skip_image = False
                # skip_file_attr = {'crowd_6-10', 'crowd_10+', 'unsure'}
                skip_file_attr = {'unsure', }
                if len(set(file_attr_dct.keys()) & skip_file_attr) > 0:
                    skip_image = True

                if len(entry['regions']) < 1:
                    skip_image = True

                if skip_image:
                    n_skipped += 1
                    continue

                # -- At this point, this anno blob *should* contain regions
                ainst_id_to_attr_anno = dict()
                # Iterate over each anno region
                for region in entry['regions'].values():
                    all_points_x = region['shape_attributes']['all_points_x']
                    all_points_y = region['shape_attributes']['all_points_y']
                    assigned_instance_id = region['assigned_instance_id']
                    # Squish x and y into [x1 y1 x2 y2 ...]
                    polygon = [z for xy_tup in zip(all_points_x, all_points_y) for z in xy_tup]
                    if assigned_instance_id in ainst_id_to_attr_anno:
                        ainst_id_to_attr_anno[assigned_instance_id].add_polygon(polygon)
                    else:
                        try:
                            this_attr_anno = AttributeAnnotation(assigned_instance_id, attr_id, [polygon, ])
                        except AssertionError:
                            print file_id, batch_filepath
                            raise
                        ainst_id_to_attr_anno[assigned_instance_id] = this_attr_anno

                if file_id in file_id_to_img_anno:
                    # Retrieve the ImageAnnotation if it was created previously
                    this_img_anno = file_id_to_img_anno[file_id]
                else:
                    this_img_anno = ImageAnnotation(file_id, img_path, h, w)
                for attr_anno in ainst_id_to_attr_anno.values():
                    this_img_anno.add_attribute_annotation(attr_anno)
                file_id_to_img_anno[file_id] = this_img_anno

    # --- Write Annotations --------------------------------------------------------------------------------------------
    anno_to_write = {'annotations': file_id_to_img_anno, 'created_at': str(datetime.datetime.now()),
                     'stats': anno_stats(file_id_to_img_anno)}
    if not osp.exists(final_out_dir):
        print '{} does not exist. Creating it...'.format(final_out_dir)
        os.makedirs(final_out_dir)
    with open(final_out_path, 'wb') as wjf:
        json.dump(anno_to_write, wjf, indent=2, cls=AnnoEncoder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=str, help="fold name", choices=['val2017', 'train2017', 'test2017'])
    parser.add_argument("snapshot_name", type=str, help="Place annotations in this snapshot directory")
    args = parser.parse_args()

    collate(args.fold, args.snapshot_name)


if __name__ == '__main__':
    main()