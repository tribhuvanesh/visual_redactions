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

# https://github.com/BVLC/caffe/issues/438
from skimage import io
io.use_plugin('matplotlib')

from PIL import Image
from scipy.misc import imread

from privacy_filters import DS_ROOT, SEG_ROOT

caffe_root = '/home/orekondy/work/opt/deeplab-v2/'
sys.path.insert(0, caffe_root + 'python')
import caffe

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

SALIENCY_DEPLOY_PATH = osp.join(SEG_ROOT, 'privacy_filters/cache/saliency/deploy.prototxt')
SALIENCY_WEIGHTS_PATH = osp.join(SEG_ROOT, 'privacy_filters/cache/saliency/weights.caffemodel')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=str, help="List of images for which to produce masks")
    parser.add_argument("outfile", type=str, help="Path to write saliency masks to (pickle file)")
    parser.add_argument("-d", "--device", type=int, choices=[0, 1, 2, 3], default=0, help="GPU device id")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    params = vars(args)

    # --- Read images --------------------------------------------------------------------------------------------------
    anno_list_file_path = osp.join(DS_ROOT, '{}2017.txt'.format(params['fold']))
    anno_path_list = map(lambda x: x.strip(), open(anno_list_file_path))
    image_id_to_path = dict()
    for anno_path in anno_path_list:
        anno = json.load(open(osp.join(DS_ROOT, anno_path)))
        image_path = osp.join(DS_ROOT, anno['image_path'])
        image_id = anno['id']
        image_id_to_path[image_id] = image_path

    # --- Setup Network ------------------------------------------------------------------------------------------------
    caffe.set_device(params['device'])
    caffe.set_mode_gpu()

    model_def = SALIENCY_DEPLOY_PATH
    model_weights = SALIENCY_WEIGHTS_PATH
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
    n_files = len(image_id_to_path)
    image_path_list = image_id_to_path.values()
    batch_size = params['batch_size']

    image_id_to_saliency = dict()

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

        for idx, f_idx in enumerate(range(start_idx, end_idx)):
            image_path = image_path_list[f_idx]
            image = caffe.io.load_image(image_path)
            transformed_image = transformer.preprocess('data', image)
            this_batch[idx] = transformed_image

        net.blobs['data'].data[...] = this_batch

        output = net.forward()
        prob = output['fc1_prob']
        # print prob.shape

        for idx, f_idx in enumerate(range(start_idx, end_idx)):
            image_path = image_path_list[f_idx]
            _, image_name = osp.split(image_path)
            image_id, _ = osp.splitext(image_name)
            sal_mask = prob[idx][1]
            assert image_id not in image_id_to_saliency
            image_id_to_saliency[image_id] = sal_mask.copy()

    # Save Saliency Masks ----------------------------------------------------------------------------------------------
    pickle.dump(image_id_to_saliency, open(params['outfile'], 'w'))


if __name__ == '__main__':
    main()
