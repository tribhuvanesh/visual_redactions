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

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import scipy
from scipy.misc import imread
from scipy.spatial import ConvexHull

from pycocotools import mask as mask_utils

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from PIL import ImageFilter

from privacy_filters.config import *
from privacy_filters.tools.common.utils import *
from privacy_filters.tools.common.extra_anno_utils import EXTRA_ANNO_PATH, bb_to_verts, load_image_id_to_text
from privacy_filters.tools.common.image_utils import bimask_to_rgba

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def refine_rle(im, pred_rle, inference_steps=40, relax_precision=False):
    bimask = mask_utils.decode(pred_rle)

    # Run on a smaller image with max(w, h) = 500
    org_w, org_h = im.size

    new_dim = 500.
    if org_w > org_h:
        new_w = new_dim
        new_h = (new_w / org_w) * org_h
    else:
        new_h = new_dim
        new_w = (new_h / org_w) * org_w
    w, h = int(new_w), int(new_h)

    _im = im.resize((w, h))
    _labels = scipy.misc.imresize(bimask, (h, w), interp='nearest')

    # Blur the image a bit so that text strokes are smoother
    _im = _im.filter(ImageFilter.GaussianBlur(radius=2))

    imarr = np.asarray(_im).copy()
    # Do not use 0s
    _labels[np.where(_labels == 0)] = 2

    # --- Dense CRF
    d = dcrf.DenseCRF2D(w, h, 2)
    U = unary_from_labels(_labels, 2, gt_prob=0.7, zero_unsure=True)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=imarr,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(inference_steps)
    map_mask = (np.argmax(Q, axis=0).reshape((h, w)) == 0).astype(np.int)

    # Resize image back to original size
    map_mask_orgsize = scipy.misc.imresize(map_mask, (org_h, org_w), interp='nearest')

    if (np.sum(map_mask_orgsize) < np.sum(bimask)) and relax_precision:
        # If we're predicting fewer pixels than before, simply return the original
        pred_rle_refined = pred_rle
    else:
        pred_rle_refined = mask_utils.encode(np.asfortranarray((map_mask_orgsize.astype(np.uint8))))

    del bimask
    del map_mask_orgsize

    return pred_rle_refined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input predictions")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    args = parser.parse_args()
    params = vars(args)

    image_index = get_image_id_info_index()

    in_predictions = json.load(open(params['infile']))
    out_predictions = []

    # Since each image has multiple predictions, but we always predict the same saliency mask, group them first
    image_id_to_predictions = defaultdict(list)
    for inpred in in_predictions:
        image_id = inpred['image_id']
        image_id_to_predictions[image_id].append(inpred)

    # --- Load Location of Text images----------------------------------------------------------------------------------
    image_id_to_textdets = dict()
    image_ids = image_id_to_predictions.keys()

    print 'Loading text regions in images...'
    for image_id in tqdm(image_ids):
        # fold = image_index[image_id]['fold']
        fold = 'test2017'
        text_anno_path = osp.join(Paths.ANNO_EXTRA_ROOT, '{}/{}.json'.format(fold, image_id))
        g_anno = json.load(open(text_anno_path))

        text_dets = []
        if 'fullTextAnnotation' in g_anno:
            # Aggregate all text blocks in this image
            for page in g_anno['fullTextAnnotation']['pages']:
                for block in page['blocks']:
                    vrts = bb_to_verts(block['boundingBox'])
                    text_dets.append(vrts)
        image_id_to_textdets[image_id] = text_dets

    # --- Process ------------------------------------------------------------------------------------------------------
    print 'Processing predictions...'
    for image_id, inpreds in tqdm(image_id_to_predictions.items()):
        image_path = image_index[image_id]['image_path']
        h, w = get_image_size(image_path)

        # What's the RLE for the text boxes in this image?
        text_dets = image_id_to_textdets[image_id]
        if len(text_dets) > 0:
            verts = None
            # Get a list of (x_i, y_i) indicating vertices of all boxes in the image
            for _det in text_dets:
                # Convert from 10x1 vector to 5x2 matrix
                det = _det.reshape([5, 2])
                if verts is None:
                    verts = det
                else:
                    verts = np.concatenate([verts, det])

            # Find the convex hull of these points
            hull = ConvexHull(verts)
            hull_x, hull_y = verts[hull.vertices, 0], verts[hull.vertices, 1]
            hull_vrts = np.concatenate([hull_x[:, None], hull_y[:, None]], axis=1)  # N x 2 matrix
            hull_vrts = np.concatenate([hull_vrts, hull_vrts[0, None]])  # Append first point to the end
            hull_vrts = np.ndarray.flatten(hull_vrts)

            # Convert to rle
            rles = mask_utils.frPyObjects([hull_vrts.tolist(), ], h, w)
            rle = mask_utils.merge(rles)

            # Perform CRF smoothing
            image_path = image_index[image_id]['image_path']
            im = Image.open(image_path)
            try:
                rle = refine_rle(im, rle, inference_steps=40, relax_precision=True)
            except ValueError:
                pass
            del im
        else:
            # Predict the entire image
            bimask = np.ones((h, w), order='F', dtype='uint8')
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