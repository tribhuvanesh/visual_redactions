#!/usr/bin/python
"""Creates data for the AMT experiment.

Given the GT file and #questions, creates:
  a. A spreadsheet with columns: img_id, attr_id, attr_question, ques_type, mask_scale, img_path
  b. Images (with/without correct/incorrect redaction)

General outline:
    for each attribute:
        for each scale:
            create question

where ques_type (given img + gt attr "A" + other gt attr) \in:
    a. scaled_mask
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

from pycocotools import mask as mask_utils

from privacy_filters.tools.common.image_utils import redact_img, scale_mask, redact_img_mask
from privacy_filters.tools.common.utils import get_image_id_info_index, load_attributes_questions, load_attributes_shorthand
from privacy_filters.tools.evaltools.evaluate_simple import SIZE_TO_ATTR_ID
from privacy_filters import SEG_ROOT
from privacy_filters.tools.evaltools.evaluate_simple import IGNORE_ATTR, MODE_TO_ATTR_ID

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

# SCALES = [0.25, 0.5, 1.0, 1.5, 2.0]
# SCALES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]
SCALES = [0.75, ]
UTIL_QUESTION = 'Is this image intelligible, so that it can be shared on social networking websites ' \
                '(e.g., Facebook, Twitter, Flickr)?'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", type=str, help="GT File")
    parser.add_argument("num_images", type=int, help="# Images per question type per attribute ID")
    parser.add_argument("out_dir", type=str, help="Where to write data")
    args = parser.parse_args()
    params = vars(args)

    # --- Create Indexes -----------------------------------------------------------------------------------------------
    image_id_index = get_image_id_info_index()
    attr_id_to_name = load_attributes_shorthand()
    attr_id_to_question = load_attributes_questions()

    gt_anno_full = json.load(open(params['gt_file']))
    gt_anno = gt_anno_full['annotations']

    #attr_ids = sorted(gt_anno_full['stats']['present_attr'])
    attr_ids = sorted(MODE_TO_ATTR_ID['textual'] + MODE_TO_ATTR_ID['visual'] + MODE_TO_ATTR_ID['multimodal'])

    image_ids = gt_anno.keys()

    np.random.seed(123)

    # Create mapping of attr_id -> [ (attr_anno + img_id + img_hw + img_path + attrs_in_img) ,... ]
    attr_id_to_details = dict()
    for image_id, entry in gt_anno.iteritems():
        image_path = entry['image_path']
        image_width, image_height = entry['image_width'], entry['image_height']
        img_attr = [x['attr_id'] for x in entry['attributes']]

        for attr_entry in entry['attributes']:
            attr_id = attr_entry['attr_id']
            if attr_id not in attr_id_to_details:
                attr_id_to_details[attr_id] = []
            # Add additional details to attribute annotation
            new_attr_entry = attr_entry
            new_attr_entry['image_id'] = image_id
            new_attr_entry['image_path'] = image_path
            new_attr_entry['image_width'] = image_width
            new_attr_entry['image_height'] = image_height
            new_attr_entry['image_attr'] = img_attr
            new_attr_entry['image_frac'] = float(attr_entry['area']) / (image_width * image_height)
            attr_id_to_details[attr_id].append(new_attr_entry)

    # --- Filter Annotations -------------------------------------------------------------------------------------------
    # Criteria to use:
    #  - Images which contain only a single instance of this type
    #  - Instance occupies < FRAC% of the image area
    #  - Do this separately for each type of attribute
    for attr_idx, attr_id in enumerate(attr_ids):
        # if attr_id in ['a39_disability_physical', 'a43_medicine', 'a7_fingerprint']:
        #     # Handle these separately due to dataset bias
        #     anno_to_use = attr_id_to_details[attr_id]
        #     np.random.shuffle(anno_to_use)
        # # elif attr_id in SIZE_TO_ATTR_ID['medium']:
        # #     anno_to_use = []
        # #     for ta in attr_id_to_details[attr_id]:
        # #         # Iterate through each instance annotation and use this image if it meets our criteria
        # #         use_this = True
        # #         # How many other instances of this attribute does the image contain?
        # #         n_other = ta['image_attr'].count(attr_id) - 1
        # #         if n_other > 0:
        # #             use_this = False
        # #         # What % of the image is this instance?
        # #         if not 0.25 <= ta['image_frac'] <= 0.5:
        # #             use_this = False
        # #         if len(ta['polygons']) < 1:
        # #             use_this = False
        # #         if use_this:
        # #             anno_to_use.append(ta)
        # #     np.random.shuffle(anno_to_use)
        # elif attr_id in SIZE_TO_ATTR_ID['small']:
        #     anno_to_use = []
        #     for ta in attr_id_to_details[attr_id]:
        #         # Iterate through each instance annotation and use this image if it meets our criteria
        #         n_other = ta['image_attr'].count(attr_id) - 1
        #         if n_other == 0:
        #             anno_to_use.append(ta)
        #     # Use everything, simply sort by large to small instances
        #     anno_to_use = sorted(anno_to_use, key=lambda x: -x['image_frac'])
        # elif attr_id in SIZE_TO_ATTR_ID['large']:
        #     # Use everything, simply sort by small to large instances
        #     anno_to_use = sorted(attr_id_to_details[attr_id], key=lambda x: x['image_frac'])
        # else:
        #     anno_to_use = attr_id_to_details[attr_id]
        #     np.random.shuffle(anno_to_use)

        anno_to_use = attr_id_to_details[attr_id]
        attr_id_to_details[attr_id] = anno_to_use

        # Filter image_ids from previous experiment
        if attr_id in ['a111_name_all', 'a24_birth_date', 'a90_email', 'a73_landmark', 'a109_person_body',
                       'a39_disability_physical', 'a106_address_current_all', 'a82_date_time', 'a8_signature',
                       'a107_address_home_all']:
            read_dir = osp.join(SEG_ROOT, 'phase7', '2017-11-01-p-vs-u-rand-3ksp-smooth-v2')
        else:
            read_dir = osp.join(SEG_ROOT, 'phase7', '2017-10-31-p-vs-u-rand-3ksp-smooth')

        image_names = os.listdir(osp.join(read_dir, 'images', 'type3', attr_id, '1.0'))
        this_image_ids = [osp.splitext(x)[0] for x in image_names]
        attr_id_to_details[attr_id] = filter(lambda x: x['image_id'] in this_image_ids, attr_id_to_details[attr_id])

        # Filter such that there's 1 image_id per attribute
        added_set = set()
        new_details_list = []
        for entry in attr_id_to_details[attr_id]:
            if entry['image_id'] not in added_set:
                new_details_list.append(entry)
                added_set.add(entry['image_id'])
        attr_id_to_details[attr_id] = new_details_list

        print '{:25s} {:5d}'.format(attr_id, len(attr_id_to_details[attr_id]))

    # --- Create Data --------------------------------------------------------------------------------------------------
    rows = []
    n_attr = len(attr_ids)
    used_img_attr = set()  # Store [(image_id, attr_id)]
    # -------------------------- For each ATTRIBUTE
    for attr_idx, attr_id in enumerate(attr_ids):
        sys.stdout.write("Processing %d/%d (%.2f%% done) %s \r" % (attr_idx, n_attr,
                                                                (attr_idx * 100.0) / n_attr, attr_id))
        sys.stdout.flush()

        # if attr_id not in (MODE_TO_ATTR_ID['textual'] + ['a39_disability_physical', 'a109_person_body', 'a8_signature']):
        #     continue
        # if attr_id == 'a49_phone':
        #     continue

        np.random.shuffle(attr_id_to_details[attr_id])

        # -------------------------- For each IMAGE
        for ques_idx in range(min(len(attr_id_to_details[attr_id]), params['num_images'])):
            anno_use = attr_id_to_details[attr_id][ques_idx]
            image_id = anno_use['image_id']
            image_path = osp.join(SEG_ROOT, anno_use['image_path'])
            im = Image.open(image_path)
            image_width, image_height = anno_use['image_width'], anno_use['image_height']

            # Get a list of all polygons of this attribute in the image
            polygons = []
            for attr_entry in gt_anno[image_id]['attributes']:
                if attr_entry['attr_id'] == attr_id:
                    if attr_id == 'a39_disability_physical':  # Handle this case slightly differently
                        this_bbox = attr_entry['bbox']
                        # Convert bbox to polygon of type [ x1 y1 x2 y2 ...]
                        x, y, w, h = this_bbox
                        this_polygon = [
                            x, y,
                            x+w, y,
                            x+w, y+h,
                            x, y+h,
                            x, y
                        ]
                        polygons += [this_polygon, ]
                    else:
                        polygons += attr_entry['polygons']
            # Get bimask for this attribute
            try:
                rles = mask_utils.frPyObjects(polygons, image_height, image_width)
            except IndexError:
                print attr_id, len(polygons)
                raise
            rle = mask_utils.merge(rles)
            bimask = mask_utils.decode(rle)

            org_bimask_scale = np.sum(bimask, dtype=np.float32) / bimask.size
            n_segments = 1000 if org_bimask_scale > 0.1 else 3000

            # -------------------------- For each SCALE
            for scale_idx, scale in enumerate(SCALES):
                scale_str = str(scale)
                # 1. Populate row
                row = dict()
                row['image_id'] = anno_use['image_id']
                row['attr_id'] = attr_id
                row['scale'] = scale
                row['attr_question'] = attr_id_to_question[attr_id]
                row['ques_type'] = 3
                row['util_question'] = UTIL_QUESTION

                # 2. Redact and save image
                img_out_dir = osp.join(params['out_dir'], 'images', 'type3', attr_id, scale_str)
                if not osp.exists(img_out_dir):
                    os.makedirs(img_out_dir)
                _, img_filename = osp.split(anno_use['image_path'])
                # Get new redaction
                if scale == 0.0:
                    scaled_bimask = np.zeros_like(bimask)
                    redacted_img = im
                elif scale == 1.0:
                    scaled_bimask = bimask
                    redacted_img = redact_img_mask(im, scaled_bimask)
                elif scale == np.inf:  # Basically, redact the entire image
                    scaled_bimask = np.ones_like(bimask)
                    redacted_img = redact_img_mask(im, scaled_bimask)
                else:
                    scaled_bimask = scale_mask(im, bimask, c=scale, n_segments=n_segments)
                    redacted_img = redact_img_mask(im, scaled_bimask)

                image_dst = osp.join(img_out_dir, img_filename)
                redacted_img.save(image_dst)
                row['image_path'] = osp.join('images', 'type3', attr_id, scale_str, img_filename)
                row['frac_redacted'] = np.sum(scaled_bimask, dtype=np.float32) / scaled_bimask.size

                rows.append(row)

    # --- Write CSV ------------------------------------------------------------------------------------------------
    csv_out_path = osp.join(params['out_dir'], 'list.csv')
    print 'Writing {} rows to {}'.format(len(rows), csv_out_path)
    with open(csv_out_path, 'w') as csvfile:
        fieldnames = sorted(rows[0]) + ['ques_idx', ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect=csv.excel)

        writer.writeheader()
        for ques_idx, row in enumerate(rows):
            row['ques_idx'] = ques_idx + 1
            writer.writerow(row)

if __name__ == '__main__':
    main()