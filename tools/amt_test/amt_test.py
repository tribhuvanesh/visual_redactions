#!/usr/bin/python
"""Creates data for the AMT experiment.

Given the GT file and #questions, creates:
  a. A spreadsheet with columns: img_id, attr_id, attr_question, ques_type, img_path, attr_id_used, gt_mask_used
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

from privacy_filters.tools.common.image_utils import redact_img
from privacy_filters.tools.common.utils import get_image_id_info_index, load_attributes_questions
from privacy_filters import SEG_ROOT

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", type=str, help="GT File")
    parser.add_argument("num_images", type=int, help="# Images per question type per attribute ID")
    parser.add_argument("out_dir", type=str, help="Where to write data")
    args = parser.parse_args()
    params = vars(args)

    # --- Create Indexes -----------------------------------------------------------------------------------------------
    image_id_index = get_image_id_info_index()
    attr_id_to_question = load_attributes_questions()

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
            if attr_id not in attr_id_to_details:
                attr_id_to_details[attr_id] = []
            # Add additional details to attribute annotation
            new_attr_entry = attr_entry
            new_attr_entry['image_id'] = image_id
            new_attr_entry['image_path'] = image_path
            new_attr_entry['image_width'] = image_width
            new_attr_entry['image_height'] = image_height
            new_attr_entry['image_attr'] = img_attr
            attr_id_to_details[attr_id].append(new_attr_entry)

    # --- Create Data --------------------------------------------------------------------------------------------------
    rows = []
    n_attr = len(attr_ids)
    used_img_attr = set()   # Store [(image_id, attr_id)]
    for attr_idx, attr_id in enumerate(attr_ids):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (attr_idx, n_attr,
                                                                (attr_idx * 100.0) / n_attr))
        sys.stdout.flush()
        # Shuffle annotations. We do this just once per attribute.
        np.random.shuffle(attr_id_to_details[attr_id])
        # --- Create Questions
        for ques_idx in range(params['num_images']):
            # Pick an annotation to use
            all_img_ids = set([x['image_id'] for x in attr_id_to_details[attr_id]])
            used_img_ids = set([x[0] for x in used_img_attr if x[1] == attr_id])
            unused_img_ids = all_img_ids - used_img_ids
            if len(unused_img_ids) > 0:
                for _anno in attr_id_to_details[attr_id]:
                    # Simply pick the first unused image
                    if _anno['image_id'] in unused_img_ids:
                        used_img_attr.add((_anno['image_id'], attr_id))
                        anno_use = _anno
                        break
            else:
                print 'Warning: Exhausted images to use for attribute {}'.format(attr_id)
                # Pick any one, doesn't matter
                anno_use = np.random.choice(attr_id_to_details[attr_id])

            # --- Type 1: [ org_img | attr_id in gt(img) ] ------------------------------------------------------------
            # Expected answer: Yes

            # 1. Populate row
            row = dict()
            row['image_id'] = anno_use['image_id']
            row['attr_id'] = attr_id
            row['attr_question'] = attr_id_to_question[attr_id]
            row['gt_attr_id'] = attr_id
            row['gt_mask_attr_id'] = None
            row['ques_type'] = 1
            row['exp'] = 'yes'

            # 2. Symlink image
            img_out_dir = osp.join(params['out_dir'], 'images', 'type1_type2', attr_id)
            if not osp.exists(img_out_dir):
                os.makedirs(img_out_dir)
            _, img_filename = osp.split(anno_use['image_path'])
            image_src = osp.join(SEG_ROOT, anno_use['image_path'])
            image_dst = osp.join(img_out_dir, img_filename)
            os.symlink(image_src, image_dst)
            row['image_path'] = osp.join('images', 'type1_type2', attr_id, img_filename)

            rows.append(row)

            # --- Type 2: [ org_img | attr_id NOT in gt(img) ]
            # Select an attribute that this image does not contain
            # Expected answer: No

            # 1. Populate row
            row = dict()
            row['image_id'] = anno_use['image_id']
            row['attr_id'] = np.random.choice(list(set(attr_ids) - set(anno_use['image_attr'])))
            row['attr_question'] = attr_id_to_question[row['attr_id']]
            row['gt_attr_id'] = attr_id
            row['gt_mask_attr_id'] = None
            row['ques_type'] = 2
            row['exp'] = 'no'

            # 2. Symlink image
            # This is the same image as that from Type 1
            # img_out_dir = osp.join(params['out_dir'], 'images', 'type2', attr_id)
            # if not osp.exists(img_out_dir):
            #     os.makedirs(img_out_dir)
            # _, img_filename = osp.split(anno_use['image_path'])
            # image_src = osp.join(SEG_ROOT, anno_use['image_path'])
            # image_dst = osp.join(img_out_dir, img_filename)
            # os.symlink(image_src, image_dst)
            row['image_path'] = osp.join('images', 'type1_type2', attr_id, img_filename)

            rows.append(row)

            # --- Type 3: [ mod_img | mask(attr_id) in gt(img) ]
            # Expected answer: No (since attribute has been redacted)

            # 1. Populate row
            row = dict()
            row['image_id'] = anno_use['image_id']
            row['attr_id'] = attr_id
            row['attr_question'] = attr_id_to_question[attr_id]
            row['gt_attr_id'] = attr_id
            row['gt_mask_attr_id'] = attr_id
            row['ques_type'] = 3
            row['exp'] = 'no'

            # 2. Redact and save image
            img_out_dir = osp.join(params['out_dir'], 'images', 'type3', attr_id)
            if not osp.exists(img_out_dir):
                os.makedirs(img_out_dir)
            _, img_filename = osp.split(anno_use['image_path'])
            image_src = osp.join(SEG_ROOT, anno_use['image_path'])
            # Get a list of all polygons of this attribute in the image
            polygons = []
            for attr_entry in gt_anno[anno_use['image_id']]['attributes']:
                if attr_entry['attr_id'] == attr_id:
                    polygons += attr_entry['polygons']
            # Redact the image
            redacted_img = redact_img(Image.open(image_src), polygons)
            image_dst = osp.join(img_out_dir, img_filename)
            redacted_img.save(image_dst)
            row['image_path'] = osp.join('images', 'type3', attr_id, img_filename)

            rows.append(row)

            # --- Type 4: [ mod_img | mask(attr_id') in gt(img); attr_id' != attr_id]
            # Expected answer: Yes (since some other random attribute has been redacted)
            row = dict()
            row['image_id'] = anno_use['image_id']
            # Select an attribute s.t.: a) is not <attr_id> b) it exists in the image c) contains a mask
            if len(anno_use['image_attr']) == 1:
                # <attr_id> is the only attribute in the image
                # Maybe draw a random polygon?
                continue
            mask_attr_id = np.random.choice(list(set(anno_use['image_attr']) - {attr_id, }))
            row['attr_id'] = mask_attr_id
            row['attr_question'] = attr_id_to_question[attr_id]
            row['gt_attr_id'] = attr_id
            row['gt_mask_attr_id'] = attr_id
            row['ques_type'] = 4
            row['exp'] = 'yes'

            # 2. Redact and save image
            img_out_dir = osp.join(params['out_dir'], 'images', 'type4', attr_id)
            if not osp.exists(img_out_dir):
                os.makedirs(img_out_dir)
            _, img_filename = osp.split(anno_use['image_path'])
            image_src = osp.join(SEG_ROOT, anno_use['image_path'])
            # Get a list of all polygons of this attribute in the image
            polygons = []
            for attr_entry in gt_anno[anno_use['image_id']]['attributes']:
                if attr_entry['attr_id'] == mask_attr_id:
                    polygons += attr_entry['polygons']
            # Redact the image
            redacted_img = redact_img(Image.open(image_src), polygons)
            image_dst = osp.join(img_out_dir, img_filename)
            redacted_img.save(image_dst)
            row['image_path'] = osp.join('images', 'type4', attr_id, img_filename)

            rows.append(row)

    # --- Write CSV ----------------------------------------------------------------------------------------------------
    csv_out_path = osp.join(params['out_dir'], 'list.csv')
    print 'Writing {} rows to {}'.format(len(rows), csv_out_path)
    with open(csv_out_path, 'w') as csvfile:
        fieldnames = sorted(rows[0]) + ['ques_idx', ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect=csv.excel)

        writer.writeheader()
        for ques_idx, row in enumerate(rows):
            row['ques_idx'] = ques_idx+1
            writer.writerow(row)


if __name__ == '__main__':
    main()