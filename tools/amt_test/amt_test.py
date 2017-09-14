#!/usr/bin/python
"""Creates data for the AMT experiment.

Given the GT file and #questions, creates:
  a. A spreadsheet with columns: img_id, attr_id, attr_question, is_fp, img_path
  b. Images (with/without correct/incorrect redaction)

General outline:
    for each attribute:
        for each ques_type:
            create question

where ques_type (given img + gt attr "A" + other gt attr) \in:
    a. org_image
      1. does A exist in img? (ANS: yes)
      2. does B \in (all-atr)\(gt-attr) in img? (ANS: no)
    b. modified_image
      1. redaction of A using GT mask
        A. does A exist in img? (ANS: no)
      2. redaction of B \in {(gt-attr)\A}
        A. does A exist in img? (ANS: yes)
      # 3. random redaction (using mask from some other image)
        # A. does A exist in img? (ANS: yes)

Notes:
  - Intention is to use TEST images for this task
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

from privacy_filters.tools.common.utils import get_image_id_info_index

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", type=str, help="GT File")
    parser.add_argument("num_images", type=str, help="# Images per question type")
    parser.add_argument("out_dir", type=str, help="Where to write data")
    args = parser.parse_args()
    params = vars(args)

    # --- Create Indexes -----------------------------------------------------------------------------------------------
    image_id_index = get_image_id_info_index()
    attr_id_to_question = dict()

    gt_anno_full = json.load(open(params['gt_file']))
    gt_anno = gt_anno_full['annotations']
    attr_ids = sorted(gt_anno_full['stats']['present_attr'])

    image_ids = gt_anno.keys()

    # Create mapping of attr_id -> [ (attr_anno + img_id + img_hw + img_path + attrs_in_img) ,... ]
    attr_id_to_details = dict()
    for image_id, entry in gt_anno.iteritems():
        image_path = entry['image_path']
        image_width, image_height = entry['image_width'], entry['image_height']
        img_attr = set([x['attr_id'] for x in entry['attributes']])

        for attr_entry in entry['attributes']:
            attr_id = attr_entry['attr_id']
            attr_id_to_details[attr_id] = attr_entry
            attr_id_to_details[attr_id]['image_path'] = image_path
            attr_id_to_details[attr_id]['image_width'] = image_width
            attr_id_to_details[attr_id]['image_height'] = image_height
            attr_id_to_details[attr_id]['image_attr'] = img_attr

    # --- Create Data --------------------------------------------------------------------------------------------------
    for attr_id in attr_ids:
        attr_ques = attr_id_to_question[attr_id]
        for ques_idx in range(params['num_images']):
            # --- Type 1: [ org_img | attr_id in gt(img) ]
            # Expected answer: Yes

            # --- Type 2: [ org_img | attr_id NOT in gt(img) ]
            # Select an image that does not contain this attribute
            # Expected answer: No

            # --- Type 3: [ mod_img | mask(attr_id) in gt(img) ]
            # Expected answer: No (since attribute has been redacted)

            # --- Type 4: [ mod_img | mask(attr_id') in gt(img); attr_id' != attr_id]
            # Expected answer: Yes (since some other random attribute has been redacted)
            pass



if __name__ == '__main__':
    main()