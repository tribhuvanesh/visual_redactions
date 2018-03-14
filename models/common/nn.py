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
from scipy.misc import imread, imresize

from pycocotools import mask as mask_utils

import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import RMSprop, SGD
from keras.utils.generic_utils import Progbar
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras import backend as K

from privacy_filters.config import *
from privacy_filters.tools.common.utils import *
from privacy_filters.tools.common.extra_anno_utils import EXTRA_ANNO_PATH, bb_to_verts

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def image_list_to_arr(image_list):
    target_img_size = (250, 250)
    n_items = len(image_list)

    X = np.zeros(shape=(n_items, target_img_size[0], target_img_size[1], 3))

    pbar = Progbar(n_items)

    for idx, (image_id, this_image_path) in enumerate(image_list):
        # ----- Image -> Mat
        resized_img_path = this_image_path.replace('images', 'images_250')
        resized_img_path = osp.join('/BS/orekondy2/work/datasets/VISPR2017', resized_img_path)

        if osp.exists(resized_img_path):
            this_image_path = resized_img_path
        else:
            this_image_path = osp.join(SEG_ROOT, this_image_path)

        img = load_img(this_image_path, target_size=target_img_size)
        img_arr = img_to_array(img)
        X[idx] = img_arr
        pbar.update(idx)

    return X


def img_to_features(X, image_list, model, batch_size=64):
    n_img, n_h, n_w, n_c = X.shape
    n_batches = n_img / batch_size + 1
    n_feat = model.output_shape[-1]

    feat_mat = np.zeros((n_img, n_feat))

    pbar = Progbar(n_batches)

    for b_idx, start_idx in enumerate(range(0, n_img, batch_size)):
        end_idx = min(start_idx + batch_size, n_img)
        this_batch_size = end_idx - start_idx

        bx = X[start_idx:end_idx]
        bx = preprocess_input(bx)
        batch_feat = model.predict(bx)

        feat_mat[start_idx:end_idx] = batch_feat
        pbar.update(b_idx)

    # Create a dict: image_id -> feat
    image_id_to_visfeat = dict()
    for i, (image_id, image_path) in enumerate(image_list):
        image_id_to_visfeat[image_id] = feat_mat[i]

    return image_id_to_visfeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    parser.add_argument("-t", "--train_mode", type=str, choices=('train', 'trainval'), default='trainval',
                        help="train/val or train+val/test")
    parser.add_argument("-d", "--device_id", type=str, help="GPU ID", default='0')
    args = parser.parse_args()
    params = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = params['device_id']
    train_mode = params['train_mode']

    # --- Load some necessary helpers ----------------------------------------------------------------------------------
    image_index = get_image_id_info_index()

    # --- Load data ----------------------------------------------------------------------------------------------------
    if train_mode == 'trainval':
        print 'train+val -> test'
        train_path = Paths.TRAINVAL_ANNO_PATH
        test_path = Paths.TEST_ANNO_PATH
    elif train_mode == 'train':
        print 'train -> val'
        train_path = Paths.TRAIN_ANNO_PATH
        test_path = Paths.VAL_ANNO_PATH
    else:
        raise ValueError('Unrecognized split')

    train_anno = json.load(open(train_path))['annotations']
    test_anno = json.load(open(test_path))['annotations']

    # --- Image -> Tensor ---------------------------------------------------------------------------------------
    train_image_list = []  # (image_id, image_path)
    test_image_list = []  # (image_id, image_path)

    for image_id, entry in train_anno.iteritems():
        train_image_list.append((image_id, entry['image_path']))

    for image_id, entry in test_anno.iteritems():
        test_image_list.append((image_id, entry['image_path']))

    train_imgarr = image_list_to_arr(train_image_list)
    print
    test_imgarr = image_list_to_arr(test_image_list)

    print
    print train_imgarr.shape
    print test_imgarr.shape

    # --- Extract Image Features ---------------------------------------------------------------------------------------
    base_resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    resnet = Model(inputs=base_resnet.input, outputs=base_resnet.output)

    x_train_img_id_to_visfeat = img_to_features(train_imgarr, train_image_list, resnet)
    print
    x_test_img_id_to_visfeat = img_to_features(test_imgarr, test_image_list, resnet)

    print
    print x_train_img_id_to_visfeat.values()[0].shape
    print x_test_img_id_to_visfeat.values()[0].shape

    # --- Predict ------------------------------------------------------------------------------------------------------
    predictions = []

    pbar = Progbar(len(test_anno))

    for idx, image_id in enumerate(test_anno.keys()):
        this_image_feat = x_test_img_id_to_visfeat[image_id]

        # What's the closest-neighbour to this image among training images?
        best_match_image_id = None
        best_match_dist = np.inf
        for train_image_id, train_feat in x_train_img_id_to_visfeat.iteritems():
            # Calc distance between features
            this_dist = np.linalg.norm(this_image_feat - train_feat, ord=2)
            if this_dist < best_match_dist:
                best_match_dist = this_dist
                best_match_image_id = train_image_id

        # Resize mask to fit this image
        test_image_id = image_id
        test_image_path = image_index[test_image_id]['image_path']
        test_im_w, test_im_h = Image.open(test_image_path).size

        train_image_id = best_match_image_id
        train_image_path = image_index[train_image_id]['image_path']
        train_im_w, train_im_h = Image.open(train_image_path).size

        # Simply predict all instances from this training image
        for attr_entry in train_anno[best_match_image_id]['attributes']:
            train_rle = attr_entry['segmentation']
            train_bimask = mask_utils.decode(train_rle)
            pred_bimask = imresize(train_bimask, (test_im_h, test_im_w), interp='nearest')
            pred_bimask = np.asfortranarray(pred_bimask)
            pred_rle = mask_utils.encode(pred_bimask)
            # pred_rle = mask_utils.merge(pred_rle, intersect=False)
            predictions.append({
                'image_id': image_id,
                'attr_id': attr_entry['attr_id'],
                'segmentation': pred_rle,
                'score': this_dist,
            })
            del train_bimask
            del pred_bimask
        pbar.update(idx)

    # Write predictions to file
    out_path = params['outfile']

    print 'Writing {} predictions tp {}'.format(len(predictions), out_path)
    json.dump(predictions, open(out_path, 'w'), indent=2)


if __name__ == '__main__':
    main()