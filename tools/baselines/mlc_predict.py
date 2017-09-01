#!/usr/bin/python
"""Returns segmentation maps from PAP model[1].

Given a trained Privacy Attribute Prediction (PAP) model[1], predicts instance segmentation masks for it.
Here, we simply provide a mask spanning the entire image if the attribute is predicted.

[1] "Towards a Visual Privacy Advisor: Understanding and Predicting Privacy Risks in Images."
Orekondy, Tribhuvanesh, Bernt Schiele, and Mario Fritz
ICCV 2017
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

# https://github.com/BVLC/caffe/issues/438
from skimage import io

from pycocotools import mask

from privacy_filters.tools.common.adapter_utils import prev_to_new_attr_vec
from privacy_filters.tools.common.utils import load_attributes, get_image_filename_index
from privacy_filters.tools.common.image_utils import get_image_size

io.use_plugin('matplotlib')

from privacy_filters import DS_ROOT, SEG_ROOT, CAFFE_ROOT

sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe/layers')  # the datalayers we will use are in this directory.
sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe')  # the tools file is in this folder

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

THRESH = 0.01


def classify_paths(net, transformer, image_file_list, batch_size=64, dataset_root=DS_ROOT):
    # Load Images ------------------------------------------------------------------------------------------------------
    image_path_list = []
    with open(image_file_list) as f:
        for line in f:
            # image_path = osp.join(dataset_root, line.strip())
            # if not osp.exists(image_path):
            #     image_path = osp.join(DS_ROOT, line.strip())
            image_path = osp.join(DS_ROOT, line.strip())
            image_path_list.append(image_path)

    n_files = len(image_path_list)

    net_img_size = net.blobs['data'].data.shape[-1]
    assert net_img_size in [224, 227, 321]

    filename_to_probs = dict()

    # Perform Forward-Passes per image ---------------------------------------------------------------------------------
    for start_idx in range(0, len(image_path_list), batch_size):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (start_idx, n_files,
                                                                (start_idx * 100.0) / n_files))
        sys.stdout.flush()

        end_idx = min(start_idx + batch_size, len(image_path_list))
        this_batch_size = end_idx - start_idx
        this_batch = np.zeros((this_batch_size, 3, net_img_size, net_img_size))

        net.blobs['data'].reshape(this_batch_size,  # batch size
                                  3,  # 3-channel (BGR) images
                                  net_img_size, net_img_size)

        # Set images
        for idx, f_idx in enumerate(range(start_idx, end_idx)):
            # image_path = image_path_list[f_idx]
            image_path = osp.join(DS_ROOT, image_path_list[f_idx])
            image_resized_path = image_path.replace('/images/', '/images_250/')
            if os.path.exists(image_resized_path):
                image_path = image_resized_path
            image = caffe.io.load_image(image_path)
            transformed_image = transformer.preprocess('data', image)
            this_batch[idx] = transformed_image

        net.blobs['data'].data[...] = this_batch

        output = net.forward()
        output_probs = output['prob']

        # Set probs per image
        for idx, f_idx in enumerate(range(start_idx, end_idx)):
            image_path = image_path_list[f_idx]
            _, filename = osp.split(image_path)
            # file_id, ext = osp.splitext(filename)
            filename_to_probs[filename] = output_probs[idx].copy()

    return filename_to_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Image Paths (example line: images/val2017/2017_25057208.jpg)")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    parser.add_argument("weights", type=str, help="Path to .caffemodel")
    parser.add_argument("deploy", type=str, help="Path to deploy.prototxt")
    parser.add_argument("-r", "--DS_ROOT", type=str, default=DS_ROOT, help="Override VISPR root")
    parser.add_argument("-d", "--device", type=int, choices=[0, 1, 2, 3], default=0, help="GPU device id")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-u", "--use_attributes", type=str, default=None, help="Use only these attributes")
    args = parser.parse_args()

    params = vars(args)

    # Initialize Network -----------------------------------------------------------------------------------------------
    caffe.set_device(params['device'])
    caffe.set_mode_gpu()

    model_def = params['deploy']
    model_weights = params['weights']

    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # Set up transformer -----------------------------------------------------------------------------------------------
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(osp.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    # mu = np.asarray([111.0, 102.0, 116.0])
    # mu = np.asarray([104.0, 117.0, 123.0])
    # print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # Classify ---------------------------------------------------------------------------------------------------------
    filename_to_probs = classify_paths(net, transformer, params['infile'], batch_size=params['batch_size'],
                                       dataset_root=params['DS_ROOT'])

    # Convert to new attribute format ----------------------------------------------------------------------------------
    # PAP was designed on 68 attributes, but now we have 74 attributes
    # So, convert previous attribute vectors to new vectors
    _, attr_id_to_idx_v1 = load_attributes(v1_attributes=True)
    idx_to_attr_id_v1= {v: k for k, v in attr_id_to_idx_v1.iteritems()}
    _, attr_id_to_idx = load_attributes()
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}

    for filename, probs in filename_to_probs.iteritems():
        if '254777' in filename:
            top_10_preds = np.argsort(-probs)[:10]
            for this_attr_idx in top_10_preds:
                print idx_to_attr_id_v1[this_attr_idx], probs[this_attr_idx]
    print

    for file_id in filename_to_probs:
        filename_to_probs[file_id] = prev_to_new_attr_vec(filename_to_probs[file_id], attr_id_to_idx_v1,
                                                          attr_id_to_idx)

    # Predict masks from attributes ------------------------------------------------------------------------------------
    # Create a mask spanning the entire image for each predicted attribute
    # Required format in: privacy_filters/tools/scripts/evaluate.py
    if params['use_attributes'] is None:
        attr_set_use = set(attr_id_to_idx.keys())
    else:
        attr_set_use = set(map(lambda x: x.strip(), open(params['use_attributes']).readlines()))

    prediction_list = []
    n_files = len(filename_to_probs)
    n_attr = len(attr_set_use)
    fname_index = get_image_filename_index()

    print 'Writing masks for {} attributes and {} files'.format(n_attr, n_files)
    for filename, probs in filename_to_probs.iteritems():
        image_path = fname_index[filename]
        file_id, ext = osp.splitext(filename)
        w, h = get_image_size(image_path)
        bimask = np.ones((h, w), order='F', dtype='uint8')
        rle = mask.encode(bimask)
        del bimask
        if '254777' in filename:
            top_10_preds = np.argsort(-probs)[:10]
            for this_attr_idx in top_10_preds:
                print idx_to_attr_id[this_attr_idx], probs[this_attr_idx]
        for this_attr_idx, this_attr_prob in enumerate(probs):
            this_attr_id = idx_to_attr_id[this_attr_idx]
            if this_attr_id in attr_set_use and this_attr_prob > THRESH:
                score_dct = {
                    'image_id': file_id,
                    'attr_id': this_attr_id,
                    'segmentation': rle,
                    'score': this_attr_prob,
                }
                prediction_list.append(score_dct)

    # Write scores -----------------------------------------------------------------------------------------------------
    with open(params['outfile'], 'w') as wf:
        json.dump(prediction_list, wf, indent=2)


if __name__ == '__main__':
    main()
