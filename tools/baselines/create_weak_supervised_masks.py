#!/usr/bin/python
"""Create Weakly supervised masks from images. Convert this to required format using: TODO

Given a CAM model, pickles the following information:
{image_id -> (localization_masks, attr_probs)}
where localization_masks = 68x41x41 matrix
      attr_probs = 68 vector
"""
import json
import time
import cPickle as pickle
import sys
import csv
import argparse
import os
import os.path as osp
import shutil

import numpy as np
import matplotlib.pyplot as plt

# https://github.com/BVLC/caffe/issues/438
from skimage import io

from privacy_filters.tools.common.image_utils import get_image_size

io.use_plugin('matplotlib')

from PIL import Image
from scipy.misc import imread

from privacy_filters import DS_ROOT, SEG_ROOT

caffe_root = '/home/orekondy/work/opt/caffe-new/'
sys.path.insert(0, caffe_root + 'python')
import caffe

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

WEAK_DEPLOY_PATH = osp.join(SEG_ROOT, 'privacy_filters/cache/weak/deploy.prototxt')
WEAK_WEIGHTS_PATH = osp.join(SEG_ROOT, 'privacy_filters/cache/weak/weights.caffemodel')

SCORE_LAYER = 'scores68'
POOL_LAYER = 'fc7_CAM'

CAM_THRESH = 0.2   # Same as used in the CAM paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=str, choices=['train', 'val', 'test'], help="fold")
    parser.add_argument("outfile", type=str, help="Path to write masks to (pickle file)")
    parser.add_argument("-d", "--device", type=int, choices=[0, 1, 2, 3], default=0, help="GPU device id")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    params = vars(args)

    # Load Images ------------------------------------------------------------------------------------------------------
    image_path_list = []
    anno_list_file_path = osp.join(DS_ROOT, '{}2017.txt'.format(params['fold']))
    anno_path_list = map(lambda x: x.strip(), open(anno_list_file_path))
    for anno_path in anno_path_list:
        anno = json.load(open(osp.join(DS_ROOT, anno_path)))
        image_path = osp.join(DS_ROOT, anno['image_path'])
        image_path_list.append(image_path)

    n_files = len(image_path_list)

    # --- Setup Network ------------------------------------------------------------------------------------------------
    caffe.set_device(params['device'])
    caffe.set_mode_gpu()

    model_def = WEAK_DEPLOY_PATH
    model_weights = WEAK_WEIGHTS_PATH
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.array([104.008, 116.669, 122.675])
    print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    net_img_size = net.blobs['data'].data.shape[-1]
    assert net_img_size in [224, 227, 321]

    # Perform Forward-Passes per batch ---------------------------------------------------------------------------------
    batch_size = params['batch_size']

    # Store in this dict: image_id -> {mask, attr_vec}
    # mask: R^{Cx41x41} mask (after CAM thresholding = CAM_THRESH)
    # attr_vec: R^68 attribute prediction probs
    image_id_to_info = dict()

    for start_idx in range(0, n_files, batch_size):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (start_idx, n_files,
                                                                (start_idx) * 100.0 / n_files))
        sys.stdout.flush()

        end_idx = min(start_idx + batch_size, n_files)
        this_batch_size = end_idx - start_idx
        this_batch = np.zeros((this_batch_size, 3, net_img_size, net_img_size))

        net.blobs['data'].reshape(this_batch_size,  # batch size
                                  3,  # 3-channel (BGR) images
                                  net_img_size, net_img_size)

        # ------------ Forward Pass for entire batch
        for idx, f_idx in enumerate(range(start_idx, end_idx)):
            image_path = image_path_list[f_idx]
            image = caffe.io.load_image(image_path)
            transformed_image = transformer.preprocess('data', image)
            this_batch[idx] = transformed_image

        net.blobs['data'].data[...] = this_batch

        output = net.forward()
        probs = net.blobs['prob'].data.copy()

        for idx, f_idx in enumerate(range(start_idx, end_idx)):
            image_path = image_path_list[f_idx]
            _, image_name = osp.split(image_path)
            image_id, _ = osp.splitext(image_name)

            # ------------ Construct CAM Map
            N, K, R, _ = net.blobs[POOL_LAYER].data.shape
            C, _K = net.params[SCORE_LAYER][0].data.shape
            assert K == _K

            CAM_params = net.params[SCORE_LAYER][0].data  # C x K
            CAM_scores = net.blobs[POOL_LAYER].data[idx, :]  # K x R X R

            heat_maps = np.zeros((C, R, R))

            for i in range(C):
                w = CAM_params[i]
                heat_maps[i, :, :] = np.sum(CAM_scores * w[:, None, None], axis=0)

            # This is a Cx41x41 mask (C=68)
            # localization_masks = heat_maps > CAM_THRESH * np.max(heat_maps)
            localization_masks = heat_maps
            attr_probs = probs[idx]

            image_id_to_info[image_id] = (localization_masks, attr_probs)

    pickle.dump(image_id_to_info, open(params['outfile'], 'w'))


if __name__ == '__main__':
    main()
