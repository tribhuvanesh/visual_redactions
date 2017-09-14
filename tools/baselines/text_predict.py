#!/usr/bin/python
"""Predict text regions as instances.

Use the collected additional annotation (from Google Cloud Vision API) to predict privacy attributes.
We predict K attributes for each text region, where the K attributes are the text-dominant attributes determined
before-hand.
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

from pycocotools import mask

from privacy_filters import DS_ROOT, SEG_ROOT
from privacy_filters.tools.common.image_utils import get_image_size
from privacy_filters.tools.common.utils import load_attributes, vec_to_labels
from privacy_filters.tools.common.adapter_utils import prev_to_new_attr_vec, prev_to_new_masks
from privacy_filters.tools.common.utils import load_attributes, get_image_id_info_index, labels_to_vec
from privacy_filters.tools.common.extra_anno_utils import EXTRA_ANNO_PATH, bb_to_verts

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


# These are the attributes whose regions are primarily text-based
TEXT_ATTR_ID = ['a107_address_home_all', 'a108_license_plate_all', 'a19_name_full', 'a20_name_first', 'a26_handwriting',
                'a73_landmark', 'a82_date_time', 'a85_username', 'a8_signature', 'a49_phone', 'a21_name_last',
                # Documents
                'a29_ausweis', 'a30_credit_card', 'a31_passport', 'a32_drivers_license', 'a33_student_id', 'a35_mail',
                'a37_receipt', 'a38_ticket'
                ]


def get_predictions(image_id, rle, attr_id_list=TEXT_ATTR_ID, score=1.0, gt_attr_id_list=TEXT_ATTR_ID):
    predictions = []
    for attr_id in (set(attr_id_list) & set(gt_attr_id_list)):
        predictions.append({
            'image_id': image_id,
            'attr_id': attr_id,
            'segmentation': rle,
            'score': score,
        })
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="List of images for which to produce masks")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    parser.add_argument("regions", type=str, choices=['block', 'paragraph', 'word', 'symbol', 'all'],
                        help="Which element to extract?")
    parser.add_argument("-g", "--gt_labels", action='store_true', default=False,
                        help="Create GT-label based predictions instead")
    args = parser.parse_args()
    params = vars(args)

    # --- Load some necessary helpers ----------------------------------------------------------------------------------
    image_id_index = get_image_id_info_index()
    _, attr_id_to_idx_v1 = load_attributes(v1_attributes=True)
    idx_to_attr_id_v1 = {v: k for k, v in attr_id_to_idx_v1.iteritems()}
    _, attr_id_to_idx = load_attributes()
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}

    # Load image_ids ---------------------------------------------------------------------------------------------------
    image_id_set = set()
    with open(params['infile']) as f:
        for _line in f:
            _, image_name = osp.split(_line.strip())
            image_id, ext = osp.splitext(image_name)
            image_id_set.add(image_id)

            # Load GT attributes too
            anno = json.load(open(image_id_index[image_id]['anno_path']))
            gt_vec_v1 = labels_to_vec(anno['labels'], attr_id_to_idx_v1)
            gt_vec = prev_to_new_attr_vec(gt_vec_v1, attr_id_to_idx_v1, attr_id_to_idx)
            image_id_index[image_id]['gt_attr_id_list'] = vec_to_labels(gt_vec, idx_to_attr_id)

    # Process these ids ------------------------------------------------------------------------------------------------
    region_to_predict = params['regions']
    predictions = []
    n_files = len(image_id_set)
    for idx, image_id in enumerate(image_id_set):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (idx, n_files,
                                                                (idx * 100.0) / n_files))
        sys.stdout.flush()

        # Load extra-annotation for this image_id
        fold = image_id_index[image_id]['fold']
        image_path = image_id_index[image_id]['image_path']
        extra_anno_path = osp.join(EXTRA_ANNO_PATH, fold, image_id + '.json')
        if params['gt_labels']:
            gt_attr_id_list = image_id_index[image_id]['gt_attr_id_list']
        else:
            gt_attr_id_list = attr_id_to_idx.keys()

        image_width, image_height = get_image_size(image_path)

        with open(extra_anno_path) as jf:
            eanno = json.load(jf)

        if 'fullTextAnnotation' not in eanno:
            continue

        for page in eanno['fullTextAnnotation']['pages']:
            page['width'] = page['width']
            page['height'] = page['height']
            # ------- Block
            for block in page['blocks']:
                block_poly = [bb_to_verts(block['boundingBox']), ]
                block_rles = mask.frPyObjects(block_poly, image_height, image_width)
                block_rle = mask.merge(block_rles)
                if region_to_predict in {'block', 'all'}:
                    predictions += get_predictions(image_id, block_rle, gt_attr_id_list=gt_attr_id_list)
                    if 'all' != region_to_predict:
                        continue
                # ------- Paragraph
                for paragraph in block['paragraphs']:
                    paragraph_poly = [bb_to_verts(paragraph['boundingBox']), ]
                    paragraph_rles = mask.frPyObjects(paragraph_poly, image_height, image_width)
                    paragraph_rle = mask.merge(paragraph_rles)
                    if region_to_predict in {'paragraph', 'all'}:
                        predictions += get_predictions(image_id, paragraph_rle, gt_attr_id_list=gt_attr_id_list)
                        if 'all' != region_to_predict:
                            continue
                    # ------- Word
                    for word in paragraph['words']:
                        word_poly = [bb_to_verts(word['boundingBox']), ]
                        word_rles = mask.frPyObjects(word_poly, image_height, image_width)
                        word_rle = mask.merge(word_rles)
                        if region_to_predict in {'word', 'all'}:
                            predictions += get_predictions(image_id, word_rle, gt_attr_id_list=gt_attr_id_list)
                            if 'all' != region_to_predict:
                                continue
                        # ------- Symbols
                        for symbol in word['symbols']:
                            symbol_poly = [bb_to_verts(symbol['boundingBox']), ]
                            symbol_rles = mask.frPyObjects(symbol_poly, image_height, image_width)
                            symbol_rle = mask.merge(symbol_rles)
                            if region_to_predict in {'symbol', 'all'}:
                                predictions += get_predictions(image_id, symbol_rle, gt_attr_id_list=gt_attr_id_list)

    # Dump predictions -------------------------------------------------------------------------------------------------
    print 'Writing {} predictions to file: {}'.format(len(predictions), params['outfile'])
    with open(params['outfile'], 'wb') as wf:
        json.dump(predictions, wf, indent=2)


if __name__ == '__main__':
    main()