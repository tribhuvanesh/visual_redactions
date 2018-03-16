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
import re
from collections import Counter
import unicodedata

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from scipy.misc import imread

import nltk
from nltk import word_tokenize

from sklearn.metrics import average_precision_score

from pycocotools import mask as mask_utils

import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import RMSprop, SGD
from keras import metrics as metrics
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import Progbar
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from privacy_filters.config import *
from privacy_filters.tools.common.utils import *
from privacy_filters.tools.common.extra_anno_utils import EXTRA_ANNO_PATH, bb_to_verts, load_image_id_to_text
from privacy_filters.tools.common.image_utils import bimask_to_rgba


__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


FNAMES_PATH = osp.join(Paths.CACHE_PATH, 'names/fnames.txt')
LNAMES_PATH = osp.join(Paths.CACHE_PATH, 'names/lnames.txt')

GEO_LOC_PATH = osp.join(Paths.CACHE_PATH, 'locations/allCountries.txt')
WIKI_LOC_PATH = osp.join(Paths.CACHE_PATH, 'locations/cities_and_countries.txt')

NAMES_SET = set()
LOCATION_SET = set()
NAME_AND_LOC = set()

STEMMER = nltk.PorterStemmer()

UNKNOWN_TOKEN = 'ukn'
UNKNOWN_IDX = 0
WORD_TO_IDX = dict()


def anno_to_data(anno_dct, attr_id_to_idx, sample_frac=1.0, mirror=False):
    target_img_size = (250, 250)
    n_items = len(anno_dct)

    attr_ids = attr_id_to_idx.keys()
    n_attr = len(attr_ids)

    # X = np.zeros(shape=(n_items, target_img_size[0], target_img_size[1], 3))
    # Y = np.zeros(shape=(n_items, n_attr))
    X = []
    Y = []
    image_id_list = []
    n_safe = 0

    pbar = Progbar(n_items)

    for idx, (image_id, entry) in enumerate(anno_dct.iteritems()):
        # ----- Labels -> Vec
        this_attr_ids = set()
        for attr_entry in entry['attributes']:
            this_attr_ids.add(attr_entry['attr_id'])

        this_attr_ids = this_attr_ids & set(attr_ids)
        if len(this_attr_ids) == 0:
            if np.random.rand() <= sample_frac:
                this_attr_ids = [SAFE_ATTR, ]
                n_safe += 1
            else:
                continue

        label_vec = np.zeros(n_attr)
        for attr_id in this_attr_ids:
            this_idx = attr_id_to_idx[attr_id]
            label_vec[this_idx] = 1
        Y.append(label_vec)

        if mirror:
            Y.append(label_vec)

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
        X.append(img_arr)

        if mirror:
            m_img = ImageOps.mirror(img)
            m_img_arr = img_to_array(m_img)
            X.append(m_img_arr)

        image_id_list.append(image_id)
        if mirror:
            image_id_list.append(image_id)

        pbar.update(idx)

    print
    print '{} / {} images contains a required label'.format(len(X) - n_safe, n_items)
    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y, image_id_list


def process_word(org_word):
    new_word = org_word.strip().lower()

    # 1. Convert numbers to tokens (replace each digit with a another token (0 in this case)
    new_word = re.sub('\d', '0', new_word)

    # 2. If word can be either name or location, use <nameloc>
    if new_word in NAME_AND_LOC:
        new_word = '<nameloc>'

    # 3. Replaces names with <name>
    if new_word in NAMES_SET:
        new_word = '<name>'

    # 4. Replace locations with <loc>
    if new_word in LOCATION_SET:
        new_word = '<loc>'

    # 5. Convert to ascii charset
    if isinstance(new_word, unicode):
        new_word = unicodedata.normalize('NFKD', new_word).encode('ascii', 'ignore')

    # 6. Stem word
    new_word = STEMMER.stem(new_word)

    return new_word


def text_to_tokens(_text):
    tokens = word_tokenize(_text.lower())
    tokens = map(process_word, tokens)

    return tokens


def get_token_list(image_id_list, image_to_text):
    token_list = []
    for image_id in image_id_list:
        if image_id in image_to_text:
            this_text = image_to_text[image_id].lower()
            this_tokens = text_to_tokens(this_text)
        else:
            this_tokens = []
        token_list += [this_tokens, ]
    return token_list


def to_idx_rep(rows, word_to_idx):
    # Convert word tokens and attributes to indexes
    n_rows = len(rows)
    n_words = len(word_to_idx)
    idx_rep = np.zeros((n_rows, n_words))
    for row_idx, row in enumerate(rows):
        for word in row:
            idx_rep[row_idx, word_to_idx.get(word, UNKNOWN_IDX)] = 1
    return idx_rep


def img_to_features(X, model, batch_size=64):
    n_img, n_h, n_w, n_c = X.shape
    n_batches = n_img / batch_size + 1
    n_feat = model.output_shape[-1]

    feat_mat = np.zeros((n_img, n_feat))

    pbar = Progbar(n_batches)

    for b_idx, start_idx in enumerate(range(0, n_img, batch_size)):
        end_idx = min(start_idx + batch_size, n_img)
        this_batch_size = end_idx - start_idx

        bx = X[start_idx:end_idx]
        bx = preprocess_input(bx.copy())
        batch_feat = model.predict(bx)

        feat_mat[start_idx:end_idx] = batch_feat.reshape((this_batch_size, n_feat))
        pbar.update(b_idx)

    return feat_mat


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
    attr_id_to_name = load_attributes_shorthand()
    image_to_text = load_image_id_to_text()

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

    # ------ Use only multimodal attributes
    attr_ids = MODE_TO_ATTR_ID['multimodal']

    # Include an ignore attribute
    SAFE_ATTR = 'a0_safe'
    attr_ids.append(SAFE_ATTR)

    n_attr = len(attr_ids)

    attr_id_to_idx = dict(zip(attr_ids, range(n_attr)))
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}
    print '# Attributes = ', n_attr
    print 'Attributes: '
    print attr_ids

    # --- Data -> Tensors ----------------------------------------------------------------------------------------------
    print 'Loading Train Data'
    (x_trainval, y_trainval, image_id_trainval) = anno_to_data(train_anno, attr_id_to_idx, sample_frac=1.0, mirror=False)
    print x_trainval.shape, y_trainval.shape

    print '\nLoading Val Data'
    (x_test, y_test, image_id_test) = anno_to_data(test_anno, attr_id_to_idx)
    print x_test.shape, y_test.shape

    # --- Set up Language part -----------------------------------------------------------------------------------------
    # ----------- Names
    for name_path in [FNAMES_PATH, LNAMES_PATH]:
        for _name in open(name_path):
            NAMES_SET.add(_name.lower().strip())

    print 'Loaded {} names...'.format(len(NAMES_SET))

    # ----------- Locations
    loc_path = WIKI_LOC_PATH

    # For more information: http://download.geonames.org/export/dump/
    with open(loc_path) as rf:
        for line in rf:
            loc = line.strip()
            LOCATION_SET.add(loc.lower())

    print 'Loaded {} locations'.format(len(LOCATION_SET))

    # ----------- Names & Locations
    NAME_AND_LOC = NAMES_SET & LOCATION_SET
    print 'Name & Loc = ', len(NAME_AND_LOC)

    x_text_trainval_tokens = get_token_list(image_id_trainval, image_to_text)
    x_text_test_tokens = get_token_list(image_id_test, image_to_text)

    # --- Tokenize -----------------------------------------------------------------------------------------------------
    VOCAB_SIZE = 1751

    word_counter = Counter()
    for row in x_text_trainval_tokens:
        for token in row:
            if len(token.strip()) >= 1:
                word_counter[token] += 1

    print 'Most common words found: ', word_counter.most_common(n=10)

    print 'Vocab size = ', len(word_counter)
    print '# >= 2 occurences = ', len(filter(lambda x: x >= 2, word_counter.values()))
    print '# >= 3 occurences = ', len(filter(lambda x: x >= 3, word_counter.values()))
    print '# >= 4 occurences = ', len(filter(lambda x: x >= 4, word_counter.values()))

    print 'Reducing vocab to size = ', VOCAB_SIZE

    # Create dict: word -> idx
    WORD_TO_IDX[UNKNOWN_TOKEN] = UNKNOWN_IDX

    for idx, (word, word_count) in enumerate(word_counter.most_common()):
        if idx >= VOCAB_SIZE - 1:
            break
        else:
            WORD_TO_IDX[word] = idx + 1

    print 'Vocab look-up dict size = ', len(WORD_TO_IDX)

    KNOWN_WORDS = set(WORD_TO_IDX.keys())

    x_text_trainval = to_idx_rep(x_text_trainval_tokens, WORD_TO_IDX)
    x_text_test = to_idx_rep(x_text_test_tokens, WORD_TO_IDX)

    print x_text_trainval.shape
    print x_text_test.shape

    # --- Extract Image Features ---------------------------------------------------------------------------------------
    base_resnet = ResNet50(include_top=False, weights='imagenet')
    resnet = Model(inputs=base_resnet.input, outputs=base_resnet.get_layer('avg_pool').output)

    x_trainval_img_feat = img_to_features(x_trainval, resnet)
    print
    x_test_img_feat = img_to_features(x_test, resnet)

    print
    print x_trainval_img_feat.shape
    print x_test_img_feat.shape

    # --- Model --------------------------------------------------------------------------------------------------------
    # ------ Define
    # -------- Vision Model
    n_feat = x_trainval_img_feat.shape[1]

    image_input = Input(shape=(n_feat,))
    encoded_image = Dense(1024, activation='relu')(image_input)
    # encoded_image = Dropout(0.2)(encoded_image)
    # encoded_image = Dense(1024, activation='relu')(encoded_image)
    encoded_image = Dropout(0.2)(encoded_image)

    # -------- Language Model
    text_input = Input(shape=(VOCAB_SIZE,), )

    x2 = Dense(512, activation='relu', input_shape=(VOCAB_SIZE,))(text_input)
    # x2 = Dropout(0.2)(x2)
    # x2 = Dense(512, activation='relu')(x2)
    encoded_text = Dropout(0.2)(x2)

    # -------- Merge Models
    merged = keras.layers.concatenate([encoded_image, encoded_text])
    merged = Dense(512, activation='relu')(merged)
    predictions = Dense(n_attr, activation='sigmoid')(merged)

    # this is the model we will train
    model = Model(inputs=[image_input, text_input], outputs=predictions)
    model.summary()

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # ------ Train
    epochs = 50
    batch_size = 128
    hist = model.fit([x_trainval_img_feat, x_text_trainval], y_trainval, epochs=epochs, batch_size=batch_size,
                     verbose=1)

    # ------ Predict
    preds = model.predict([x_test_img_feat, x_text_test], verbose=1)
    ap_scores = average_precision_score(y_test, preds, average=None)

    attr_id_score = zip(attr_ids, ap_scores)

    print
    for attr_id, attr_score in sorted(attr_id_score, key=lambda x: -x[1]):
        print '{:>20s}: {:.3f}'.format(attr_id_to_name[attr_id], attr_score)

    print 'C-MAP = ', np.mean(ap_scores)

    # --- Write Output -------------------------------------------------------------------------------------------------
    out_path = params['outfile']
    predictions = []

    n_test_rows = preds.shape[0]
    thresh = 0.0

    for row_idx in range(n_test_rows):
        this_region_probs = preds[row_idx]
        pred_idxs = np.where(this_region_probs >= thresh)[0]
        image_id = image_id_test[row_idx]
        h, w = test_anno[image_id]['image_height'], test_anno[image_id]['image_width']
        bimask = np.ones((h, w), order='F', dtype='uint8')
        rle = mask_utils.encode(bimask)
        del bimask
        for pred_idx in pred_idxs:
            attr_id = idx_to_attr_id[pred_idx]
            if attr_id is not SAFE_ATTR:
                predictions.append({
                    'image_id': image_id_test[row_idx],
                    'attr_id': attr_id,
                    'segmentation': rle,
                    'score': this_region_probs[pred_idx].astype(float),
                })

    print
    print 'Writing {} predictions to {}'.format(len(predictions), out_path)
    json.dump(predictions, open(out_path, 'wb'), indent=2)


if __name__ == '__main__':
    main()