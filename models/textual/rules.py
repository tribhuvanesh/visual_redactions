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

from pycocotools import mask as mask_utils

from privacy_filters.config import *
from privacy_filters.tools.common.utils import *

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_proxy", type=str, help="Path to Test proxy file")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    args = parser.parse_args()
    params = vars(args)

    # --- Load some necessary helpers ----------------------------------------------------------------------------------
    image_index = get_image_id_info_index()
    attr_id_to_name = load_attributes_shorthand()

    # --- Load data ----------------------------------------------------------------------------------------------------
    val_anno = json.load(open(params['test_proxy']))

    print '# Test images = ', len(val_anno)

    attr_ids = [SAFE_ATTR, ] + MODE_TO_ATTR_ID['textual']
    n_attr = len(attr_ids)

    attr_id_to_idx = dict(zip(attr_ids, range(n_attr)))
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}
    print '# Attributes = ', n_attr
    print 'Attributes: '

    # --- Load Priors --------------------------------------------------------------------------------------------------
    # ------ Names
    fnames_path = '/BS/orekondy2/work/privacy_filters/cache/names/fnames.txt'
    lnames_path = '/BS/orekondy2/work/privacy_filters/cache/names/lnames.txt'

    names_set = set()

    for name_path in [fnames_path, lnames_path]:
        for _name in open(name_path):
            names_set.add(_name.lower().strip())

    print 'Loaded {} names...'.format(len(names_set))

    # ------ Locations
    # Option A --- GeoNames
    # loc_path = '/BS/orekondy2/work/privacy_filters/cache/locations/allCountries.txt'
    # location_set = set()
    #
    # # # For more information: http://download.geonames.org/export/dump/
    # with open(loc_path) as rf:
    #     for line in rf:
    #         tokens = line.strip().split('\t')
    #         asciiname = tokens[2]
    #
    #         # Use only names of (A): country, state, region or (P): city, village, etc
    #         feature_class = tokens[6]
    #         if feature_class in ['A', 'P']:
    #             location_set.add(asciiname.lower())

    # Option B --- Wiki Names
    loc_path = '/BS/orekondy2/work/privacy_filters/cache/locations/cities_and_countries.txt'
    location_set = set()

    with open(loc_path) as rf:
        for line in rf:
            loc = line.strip()
            location_set.add(loc.lower())

    print 'Loaded {} locations'.format(len(location_set))

    # --- Rules --------------------------------------------------------------------------------------------------------
    if_name_attrs = [
        u'a111_name_all',
    ]

    if_digit_attrs = [
        u'a49_phone',
        u'a82_date_time',
        u'a24_birth_date',
    ]

    if_loc_attrs = ["a106_address_current_all",
                    "a107_address_home_all",
                    "a73_landmark", ]

    if_email_attrs = [u'a90_email',
                      ]

    # --- Predict ------------------------------------------------------------------------------------------------------
    predictions = []

    for image_id, entry in val_anno.iteritems():
        sequence = entry['sequence']
        for idx, (_word, word_rle) in enumerate(zip(sequence['word_text'], sequence['word_rle'])):
            word = _word.lower().strip()

            # --- Name
            if word in names_set:
                for attr_id in if_name_attrs:
                    predictions.append({
                        'image_id': image_id,
                        'attr_id': attr_id,
                        'segmentation': word_rle,
                        'score': 1.0,
                        'text': word,
                    })

            # --- Location
            if word in location_set:
                for attr_id in if_loc_attrs:
                    predictions.append({
                        'image_id': image_id,
                        'attr_id': attr_id,
                        'segmentation': word_rle,
                        'score': 1.0,
                        'text': word,
                    })

            # --- Number
            if any(char.isdigit() for char in word):
                for attr_id in if_digit_attrs:
                    predictions.append({
                        'image_id': image_id,
                        'attr_id': attr_id,
                        'segmentation': word_rle,
                        'score': 1.0,
                        'text': word,
                    })

            # --- Email address
            if '@' in word:
                # Expect format: <foo> @ <bar> <.> <com>
                for attr_id in if_email_attrs:
                    context_rle = [sequence['word_rle'][j] for j in [idx - 1, idx, idx + 1, idx + 2]]
                    context_rle = filter(lambda x: x is not None, context_rle)
                    merged_rle = mask_utils.merge(context_rle)

                    predictions.append({
                        'image_id': image_id,
                        'attr_id': attr_id,
                        'segmentation': merged_rle,
                        'score': 1.0,
                        'text': word,
                    })

    # --- Write Output -------------------------------------------------------------------------------------------------
    outfile = params['outfile']
    print 'Writing {} predictions to {}'.format(len(predictions), outfile)
    with open(outfile, 'w') as wf:
        json.dump(predictions, wf, indent=2)


if __name__ == '__main__':
    main()