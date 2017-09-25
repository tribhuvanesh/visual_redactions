#!/usr/bin/python
"""Retrains a multi-label classifier and predicts entire image.

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

from pycocotools import mask as mask_utils

from collections import Counter

from privacy_filters.tools.common.utils import *
from privacy_filters.tools.common.extra_anno_utils import EXTRA_ANNO_PATH, bb_to_verts
from privacy_filters import SEG_ROOT

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Input
from keras.optimizers import RMSprop, SGD
from keras import metrics as metrics
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.generic_utils import Progbar
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from sklearn.metrics import average_precision_score

sys.path.insert(0, SEG_ROOT)

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def anno_to_data(anno_dct, attr_id_to_idx, target_img_size=(250, 250)):
    n_items = len(anno_dct)
    n_attr = len(attr_id_to_idx)

    X = np.zeros(shape=(n_items, target_img_size[0], target_img_size[1], 3))
    Y = np.zeros(shape=(n_items, n_attr))
    image_id_list = []

    pbar = Progbar(n_items)

    for idx, (image_id, entry) in enumerate(anno_dct.iteritems()):
        # ----- Labels -> Vec
        this_attr_ids = set()
        for attr_entry in entry['attributes']:
            this_attr_ids.add(attr_entry['attr_id'])
        label_vec = np.zeros(n_attr)
        for attr_id in this_attr_ids:
            this_idx = attr_id_to_idx[attr_id]
            label_vec[this_idx] = 1
        Y[idx] = label_vec

        # ----- Image -> Mat
        this_image_path = entry['image_path']

        resized_img_path = this_image_path.replace('images', 'images_250')
        resized_img_path = osp.join('/BS/orekondy2/work/datasets/VISPR2017', resized_img_path)

        if osp.exists(resized_img_path):
            this_image_path = resized_img_path
        else:
            this_image_path = osp.join(SEG_ROOT, this_image_path)

        img = load_img(this_image_path, target_size=target_img_size)
        img_arr = img_to_array(img)
        X[idx] = img_arr

        image_id_list.append(image_id)

        pbar.update(idx)

    return X, Y, image_id_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expt_dir", type=str, help="Path containing (train, val, test).json")
    parser.add_argument("out_prefix", type=str, help="Prefix for writing model and prediction")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="# Epochs")
    parser.add_argument("-i", "--image_size", type=int, default=250, help="Image Size NxN")
    parser.add_argument("-m", "--model", type=str, choices=['inception', 'resnet'], default='resnet', help="Image Model")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-d", "--device", type=int, default=0, help="GPU device")
    parser.add_argument("-r", "--retrain", default=False, action='store_true',
                        help="Retrains all layers (instead of last few layers")
    parser.add_argument("-a", "--augment", type=int, default=0,
                        help="Augment data s.t. each attribute appears at least these many times")
    parser.add_argument("-c", "--class_weight", default=False, action='store_true',
                        help="Use class weights during training")
    args = parser.parse_args()

    params = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(params['device'])

    # Load Data --------------------------------------------------------------------------------------------------------
    train_path = osp.join(SEG_ROOT, 'annotations', params['expt_dir'], 'train2017.json')
    val_path = osp.join(SEG_ROOT, 'annotations', params['expt_dir'], 'val2017.json')

    train_anno_full = json.load(open(train_path))
    train_anno = train_anno_full['annotations']
    val_anno_full = json.load(open(val_path))
    val_anno = val_anno_full['annotations']

    anno_full = dict(train_anno.items() + val_anno.items())

    print '# Train images = ', len(train_anno)
    print '# Val images = ', len(val_anno)
    print '# ALL images = ', len(anno_full)

    # Helpers  ---------------------------------------------------------------------------------------------------------
    image_index = get_image_id_info_index()
    attr_id_to_name = load_attributes_shorthand()
    image_ids = sorted(train_anno.keys())

    attr_ids_set = set()
    for img_id, entry in anno_full.iteritems():
        for attr_entry in entry['attributes']:
            attr_ids_set.add(attr_entry['attr_id'])
    attr_ids = sorted(attr_ids_set)
    attr_names = [attr_id_to_name[attr_id] for attr_id in attr_ids]

    n_img = len(image_ids)
    n_attr = len(attr_ids)

    attr_id_to_idx = dict(zip(attr_ids, range(n_attr)))
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}

    print '# Images = ', n_img
    print '# Attributes = ', n_attr
    print 'Attributes: '
    print attr_ids

    target_image_size = (params['image_size'], params['image_size'])
    print 'Loading Train Data'
    (x_train, y_train, image_id_train) = anno_to_data(train_anno, attr_id_to_idx, target_image_size)
    print '\nLoading Val Data'
    (x_val, y_val, image_id_val) = anno_to_data(val_anno, attr_id_to_idx, target_image_size)

    # Data Augmentation ------------------------------------------------------------------------------------------------
    if params['augment'] > 0:
        # Current Stats
        counter = Counter()
        anno_dct = train_anno
        for idx, (image_id, entry) in enumerate(anno_dct.iteritems()):
            this_attr_ids = set()
            for attr_entry in entry['attributes']:
                this_attr_ids.add(attr_entry['attr_id'])
            for attr_id in this_attr_ids:
                counter[attr_id] += 1

        min_count = params['augment']  # Each attribute should appear at least these many times
        x_train_aug, y_train_aug = [], []

        # Which attributes do we need to augment for?
        attr_ids_aug = filter(lambda x: counter[x] < min_count, attr_ids)
        print 'Augmenting data to {} for {} attributes:\n{}'.format(min_count, len(attr_ids_aug), attr_ids_aug)

        for attr_id in attr_ids_aug:
            num_remaining = min_count - counter[attr_id]
            attr_idx = attr_id_to_idx[attr_id]
            all_labels = y_train[:, attr_idx]  # Get labels for all images
            train_idxs = np.where(all_labels > 0)[0]  # Images which contains this attribute

            for i in range(num_remaining):
                # Select a random image from x_train
                row_idx = np.random.choice(train_idxs)

                # Add to augmented set
                x_train_aug.append(x_train[row_idx])
                y_train_aug.append(y_train[row_idx])

        x_train_aug = np.asarray(x_train_aug)
        y_train_aug = np.asarray(y_train_aug)

        print '# Augmented Rows = ', x_train_aug.shape[0]

        x_train = np.concatenate((x_train, x_train_aug), axis=0)
        y_train = np.concatenate((y_train, y_train_aug), axis=0)

    # Class Weights ----------------------------------------------------------------------------------------------------
    class_weight = np.ones((n_attr, ))
    if params['class_weight']:
        n_samples, n_classes = y_train.shape
        for i in range(n_attr):
            class_weight[i] = n_samples / (n_classes * np.sum(y_train[:, i]))

    # Image Preprocessing  ---------------------------------------------------------------------------------------------
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(x_train)

    # Model ------------------------------------------------------------------------------------------------------------
    if params['model'] == 'inception':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif params['model'] == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False)
    else:
        raise ValueError('Unrecognized model')

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_attr, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    if not params['retrain']:
        for layer in base_model.layers:
            layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    # train the model on the new data for a few epochs
    epochs = params['epochs']
    batch_size = params['batch_size']
    print 'Training with #Epochs = {}, Batch size = {}'.format(epochs, batch_size)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                        class_weight=class_weight)

    model_out_path = params['out_prefix'] + '.h5'
    model.save(model_out_path)

    # Evaluate ---------------------------------------------------------------------------------------------------------
    print
    print 'Evaluating Model...'
    preds = model.predict(x_val, verbose=1)

    ap_scores = average_precision_score(y_val, preds, average=None)

    attr_id_score = zip(attr_ids, ap_scores)

    for attr_id, attr_score in sorted(attr_id_score, key=lambda x: -x[1]):
        print '{:>20s}: {:.3f}'.format(attr_id_to_name[attr_id], attr_score)

    print 'C-MAP = ', np.mean(ap_scores)

    # Write Predictions ------------------------------------------------------------------------------------------------
    out_path = params['out_prefix'] + '.json'
    predictions = []

    n_val_rows = preds.shape[0]
    thresh = 0.1

    for row_idx in range(n_val_rows):
        this_region_probs = preds[row_idx]
        pred_idxs = np.where(this_region_probs >= thresh)[0]
        image_id = image_id_val[row_idx]
        h, w = anno_full[image_id]['image_height'], anno_full[image_id]['image_width']
        bimask = np.ones((h, w), order='F', dtype='uint8')
        rle = mask_utils.encode(bimask)
        del bimask
        for pred_idx in pred_idxs:
            attr_id = idx_to_attr_id[pred_idx]
            predictions.append({
                'image_id': image_id_val[row_idx],
                'attr_id': attr_id,
                'segmentation': rle,
                'score': this_region_probs[pred_idx].astype(float),
            })

    print 'Writing {} predictions'.format(len(predictions))
    json.dump(predictions, open(out_path, 'w'), indent=2)


if __name__ == '__main__':
    main()