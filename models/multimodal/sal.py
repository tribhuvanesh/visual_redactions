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

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import scipy
from scipy.misc import imread

from pycocotools import mask as mask_utils

from privacy_filters.config import *
from privacy_filters.tools.common.utils import *
from privacy_filters.tools.common.extra_anno_utils import EXTRA_ANNO_PATH, bb_to_verts, load_image_id_to_text
from privacy_filters.tools.common.image_utils import bimask_to_rgba

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


SALIENCY_THRESH = 0.8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input predictions")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    args = parser.parse_args()
    params = vars(args)

    image_index = get_image_id_info_index()

    # Load Saliency Mask
    print 'Loading Saliency Mask...'
    image_id_to_saliency = pickle.load(open(osp.join(Paths.CACHE_PATH, 'saliency', 'test.pkl')))

    in_predictions = json.load(open(params['infile']))
    out_predictions = []

    # Since each image has multiple predictions, but we always predict the same saliency mask, group them first
    image_id_to_predictions = defaultdict(list)
    for inpred in in_predictions:
        image_id = inpred['image_id']
        image_id_to_predictions[image_id].append(inpred)

    for image_id, inpreds in image_id_to_predictions.iteritems():
        image_path = image_index[image_id]['image_path']
        h, w = get_image_size(image_path)

        # Predict salient pixels as attribute
        lowres_saliency = image_id_to_saliency[image_id]
        # Resize mask (originally is 321 x 321)
        highres_saliency = scipy.misc.imresize(lowres_saliency, [h, w], interp='bilinear', mode='F')
        # Binarize it
        bimask = (highres_saliency > (SALIENCY_THRESH * np.max(highres_saliency))).astype('uint8')
        bimask = np.asfortranarray(bimask)
        del lowres_saliency
        del highres_saliency
        rle = mask_utils.encode(bimask)
        del bimask

        for inpred in inpreds:
            image_id = inpred['image_id']
            attr_id = inpred['attr_id']
            score = inpred['score']

            out_predictions.append({
                'image_id': image_id,
                'attr_id': attr_id,
                'segmentation': rle,
                'score': score
            })

    print
    out_path = params['outfile']
    print 'Writing {} predictions to {}'.format(len(out_predictions), out_path)
    json.dump(out_predictions, open(out_path, 'wb'), indent=2)


if __name__ == '__main__':
    main()