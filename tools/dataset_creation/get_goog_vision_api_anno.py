#!/usr/bin/python
"""Obtain additional annotations using Google Cloud Vision API.

Get the following annotations using the Cloud Vision API:
- LABEL_DETECTION
- TEXT_DETECTION
- DOCUMENT_TEXT_DETECTION
- LANDMARK_DETECTION
- FACE_DETECTION
- SAFE_SEARCH_DETECTION

Additionally, this API only supports images <4mb. However, many of VISPR images exceed this limit.
So, this script surrogates images with downscaled ones when possible. It additionally extrapolates locations
to match original size.
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

import requests
import io
import base64
from copy import deepcopy

from privacy_filters.tools.common.image_utils import get_image_size
from privacy_filters.tools.common.secret import GOOGLE_API_KEY
from privacy_filters import DS_ROOT, SEG_ROOT

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

DETECTION_TYPES = [
    'TEXT_DETECTION',
    'DOCUMENT_TEXT_DETECTION',
    'LABEL_DETECTION',
    'LANDMARK_DETECTION',
    'SAFE_SEARCH_DETECTION',
    'FACE_DETECTION'
]

GOOGLE_VPI_URL = 'https://vision.googleapis.com/v1/images:annotate?key={}'.format(GOOGLE_API_KEY)


def encode_image(image_path):
    with io.open(image_path, 'rb') as image_file:
        image_content = image_file.read()
    return base64.b64encode(image_content)


def get_annotation(image_path, detection_types=DETECTION_TYPES):
    request_dct = {
        "requests": [
            {
                "image": {
                    "content": encode_image(image_path)
                },
                "features": [
                    # Example: 
                    # {
                    #     "type": "TEXT_DETECTION"
                    # },
                ]
            }
        ]
    }
    # Add features
    for det_type in detection_types:
        request_dct['requests'][0]['features'].append({'type': det_type})
    request_json = json.dumps(request_dct)
    response = requests.post(url=GOOGLE_VPI_URL, data=request_json, headers={'Content-Type': 'application/json'})
    response_json = response.text
    return response_json


def fix_sizes(vdct, org_im_size, cur_im_size):
    """
    Images are sometimes downsized, since API rejects images > 4mb (and is also slower for large images)
    So, given response by Google Cloud Vision API and the original image size, this fixes any location information
    to match the original image's resolution.
    Documentation of Response types:
    https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate?authuser=1#annotateimageresponse
    :param vdct:
    :param org_im_size:
    :return:
    """
    org_w, org_h = org_im_size
    cur_w, cur_h = cur_im_size

    # w/h multiplicative factor. Need to multiply any location factor by this amount
    w_scale = org_w / float(cur_w)
    h_scale = org_h / float(cur_h)

    if 'textAnnotations' in vdct:
        vdct['textAnnotations_prev'] = deepcopy(vdct['textAnnotations'])
        vdct['textAnnotations'] = fix_entity_anno(vdct['textAnnotations'], w_scale, h_scale)
    if 'fullTextAnnotation' in vdct:
        vdct['fullTextAnnotation_prev'] = deepcopy(vdct['fullTextAnnotation'])
        vdct['fullTextAnnotation'] = fix_full_text_anno(vdct['fullTextAnnotation'], w_scale, h_scale)
    if 'landmarkAnnotations' in vdct:
        vdct['landmarkAnnotations_prev'] = deepcopy(vdct['landmarkAnnotations'])
        vdct['landmarkAnnotations'] = fix_entity_anno(vdct['landmarkAnnotations'], w_scale, h_scale)
    if 'faceAnnotations' in vdct:
        vdct['faceAnnotations_prev'] = deepcopy(vdct['faceAnnotations'])
        vdct['faceAnnotations'] = fix_face_entity_anno(vdct['faceAnnotations'], w_scale, h_scale)

    return vdct


def fix_entity_anno_entry(entry, w_scale, h_scale):
    """
    Fix EntityAnnotation
    https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate?authuser=1#EntityAnnotation
    :param anno:
    :param w_scale:
    :param h_scale:
    :return:
    """
    if 'boundingPoly' in entry:
        if 'vertices' in entry['boundingPoly']:
                for vrt in entry['boundingPoly']['vertices']:
                    if 'x' in vrt:
                        vrt['x'] = int(np.round(vrt['x'] * w_scale))
                    else:
                        print 'Warning: x missing in boundingPoly'
                    if 'y' in vrt:
                        vrt['y'] = int(np.round(vrt['y'] * h_scale))
                    else:
                        print 'Warning: y missing in boundingPoly'
    return entry


def fix_face_entity_anno_entry(entry, w_scale, h_scale):
    """
    Fix EntityAnnotation
    https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate?authuser=1#EntityAnnotation
    :param anno:
    :param w_scale:
    :param h_scale:
    :return:
    """
    if 'boundingPoly' in entry:
        entry['boundingPoly'] = fix_bounding_poly(entry['boundingPoly'], w_scale, h_scale)
    if 'fdBoundingPoly' in entry:
        entry['fdBoundingPoly'] = fix_bounding_poly(entry['fdBoundingPoly'], w_scale, h_scale)
    if 'landmarks' in entry:
        for landmark_entry in entry['landmarks']:
            if 'x' in landmark_entry['position']:
                landmark_entry['position']['x'] = landmark_entry['position']['x'] * float(w_scale)
            if 'y' in landmark_entry['position']:
                landmark_entry['position']['y'] = landmark_entry['position']['y'] * float(h_scale)
    return entry


def fix_entity_anno(anno_list, w_scale, h_scale):
    """
    The text/landmark annotation object is simply a list of entities.
    Fix Text Annotations (not Full Text!). Return a new fixed annotation object.
    :type anno_list: list
    :param anno_list:
    :param w_scale:
    :param h_scale:
    :return:
    """
    new_anno_list = []
    for idx in range(len(anno_list)):
        this_entry = deepcopy(anno_list[idx])
        this_entry = fix_entity_anno_entry(this_entry, w_scale, h_scale)
        new_anno_list.append(this_entry)
    return new_anno_list


def fix_face_entity_anno(anno_list, w_scale, h_scale):
    """
    The Face annotation object is simply a list of entities.
    https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate#FaceAnnotation
    :type anno_list: list
    :param anno_list:
    :param w_scale:
    :param h_scale:
    :return:
    """
    new_anno_list = []
    for idx in range(len(anno_list)):
        this_entry = deepcopy(anno_list[idx])
        this_entry = fix_face_entity_anno_entry(this_entry, w_scale, h_scale)
        new_anno_list.append(this_entry)
    return new_anno_list


def fix_bounding_poly(bounding_poly, w_scale, h_scale):
    for vrt in bounding_poly['vertices']:
        if 'x' in vrt:
            vrt['x'] = int(np.round(vrt['x'] * w_scale))
        else:
            print 'Warning: x missing in boundingPoly'
        if 'y' in vrt:
            vrt['y'] = int(np.round(vrt['y'] * h_scale))
        else:
            print 'Warning: y missing in boundingPoly'
    return bounding_poly


def fix_full_text_anno(_entry, w_scale, h_scale):
    """
    Fix TextAnnotation:
    https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate?authuser=1#TextAnnotation
    Returns a new modified copy.
    :param _entry:
    :param w_scale:
    :param h_scale:
    :return:
    """
    entry = deepcopy(_entry)
    for page in entry['pages']:
        page['width'] = int(np.round(w_scale * page['width']))
        page['height'] = int(np.round(h_scale * page['height']))
        for block in page['blocks']:
            block['boundingBox'] = fix_bounding_poly(block['boundingBox'], w_scale, h_scale)
            for paragraph in block['paragraphs']:
                paragraph['boundingBox'] = fix_bounding_poly(paragraph['boundingBox'], w_scale, h_scale)
                for word in paragraph['words']:
                    word['boundingBox'] = fix_bounding_poly(word['boundingBox'], w_scale, h_scale)
                    for symbol in word['symbols']:
                        symbol['boundingBox'] = fix_bounding_poly(symbol['boundingBox'], w_scale, h_scale)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anno_list", type=str, help="Path to Annotation file (from VISPR-2017)")
    parser.add_argument("red_dir", type=str, help="Looks for reduced file-size images in this directory")
    parser.add_argument("output_dir", type=str, help="Places annotation files in this directory as <filename>.json")
    parser.add_argument("-i", "--skip_if_exists", action='store_true', default=True, help="Skip image if it exists")
    args = parser.parse_args()

    params = vars(args)

    anno_files = [x.strip() for x in open(params['anno_list']).readlines()]
    print '# Files to process = ', len(anno_files)

    n_processed = 0
    n_skipped = 0

    for idx, line in enumerate(anno_files):
        _, file_id_ext = osp.split(line)
        file_id, _ = osp.splitext(file_id_ext)

        out_path = osp.join(params['output_dir'], file_id + '-extra.json')
        if osp.exists(out_path) and params['skip_if_exists']:
            n_skipped += 1
            continue

        sys.stdout.write("Processing %d/%d (%.2f%% done)   \r" % (idx, len(anno_files),
                                                                  idx * 100.0 / len(anno_files)))
        sys.stdout.flush()

        # ------------- Get Image Path
        anno_path = osp.abspath(osp.join(DS_ROOT, line))
        with open(anno_path) as jf:
            anno = json.load(jf)
        img_path_partial = anno['image_path']
        _, img_filename = osp.split(img_path_partial)
        # Check if a reduced version of this image exists
        org_img_path = osp.abspath(osp.join(DS_ROOT, img_path_partial))
        org_w, org_h = get_image_size(org_img_path)
        reduced_img_path = osp.join(params['red_dir'], img_filename)
        if osp.exists(reduced_img_path):
            image_path = osp.abspath(reduced_img_path)
            new_w, new_h = get_image_size(image_path)
        else:
            image_path = org_img_path
            new_w, new_h = org_w, org_h

        # ------------- Process Response
        vision_response_json = get_annotation(image_path)
        vision_response_dct = json.loads(vision_response_json)
        if len(vision_response_dct['responses']) != 1:
            print 'WARNING: Got {} responses for request: {}'.format(len(vision_response_dct['responses']), file_id)
        # Only use the first response (which is always the case since we query only a single image)
        vision_response_dct = vision_response_dct['responses'][0]

        # Add some additional details
        vision_response_dct['file_info'] = {
            'image_path_prev': image_path,
            'img_size_prev': (new_w, new_h),

            'image_path': org_img_path,
            'size': (org_w, org_h),

            'anno_path': anno_path,
        }

        if org_img_path != image_path:
            vision_response_dct = fix_sizes(vision_response_dct, (org_w, org_h), (new_w, new_h))

        # ------------- Write Response
        with open(out_path, 'wb') as wf:
            json.dump(vision_response_dct, wf, indent=2)

        n_processed += 1

    print '# Processed: ', n_processed
    print '# Skipped: ', n_skipped


if __name__ == '__main__':
    main()
