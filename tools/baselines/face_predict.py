#!/usr/bin/python
"""Use extra annotations (Google Cloud Vision API) to predict faces.

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

from pycocotools import mask

from privacy_filters import DS_ROOT, SEG_ROOT
from privacy_filters.tools.common.image_utils import get_image_size
from privacy_filters.tools.common.utils import load_attributes
from privacy_filters.tools.common.adapter_utils import prev_to_new_attr_vec, prev_to_new_masks
from privacy_filters.tools.common.utils import load_attributes, get_image_id_info_index, labels_to_vec

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

EXTRA_ANNO_PATH = osp.join(SEG_ROOT, 'annotations-extra-2')


def bb_to_verts(bb):
    vrts_dct = bb['vertices']  # List of [ {x:__, y:__}, ... ]
    vrts = [[d.get('x', 0), d.get('y', 0)] for d in vrts_dct]  # Convert to [ [x1, y1], [x2, y2], ..]
    vrts.append(vrts[0])  # Reconnect to first vertex
    return np.ndarray.flatten(np.asarray(vrts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="List of images for which to produce masks")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    args = parser.parse_args()
    params = vars(args)

    # --- Load some necessary helpers ----------------------------------------------------------------------------------
    image_id_index = get_image_id_info_index()
    attr_id_to_name, attr_id_to_idx = load_attributes()

    # Load image_ids ---------------------------------------------------------------------------------------------------
    image_id_set = set()
    with open(params['infile']) as f:
        for _line in f:
            _, image_name = osp.split(_line.strip())
            image_id, ext = osp.splitext(image_name)
            image_id_set.add(image_id)

    # Process these ids ------------------------------------------------------------------------------------------------
    predictions = []
    n_files = len(image_id_set)
    for idx, image_id in enumerate(image_id_set):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (idx, n_files,
                                                                (idx * 100.0) / n_files))
        sys.stdout.flush()

        # Load extra-annotation for this image_id
        fold = image_id_index[image_id]['fold']
        image_path = image_id_index[image_id]['image_path']
        extra_anno_path = osp.join(EXTRA_ANNO_PATH, fold, image_id + '-extra.json')

        image_width, image_height = get_image_size(image_path)

        with open(extra_anno_path) as jf:
            eanno = json.load(jf)

        for face_entry in eanno.get('faceAnnotations', []):
            this_poly = [bb_to_verts(face_entry['fdBoundingPoly']), ]
            rles = mask.frPyObjects(this_poly, image_height, image_width)
            rle = mask.merge(rles)

            predictions.append({
                'image_id': image_id,
                'attr_id': 'a105_face_all',
                'segmentation': rle,
                'score': face_entry['detectionConfidence'],
            })

    # Dump predictions -------------------------------------------------------------------------------------------------
    print 'Writing {} predictions to file: {}'.format(len(predictions), params['outfile'])
    with open(params['outfile'], 'wb') as wf:
        json.dump(predictions, wf, indent=2)


if __name__ == '__main__':
    main()