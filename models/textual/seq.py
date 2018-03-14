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
from scipy.misc import imread

import unicodedata
import re
from collections import Counter

from scipy.misc import imresize
from scipy import ndimage

from pycocotools import mask as mask_utils

import keras
from keras.models import Sequential, Model
from keras.layers import *
# from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, SGD
from keras import metrics as metrics
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import Progbar
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

import nltk

from privacy_filters.config import *
from privacy_filters.tools.common.utils import *
from privacy_filters.tools.common.image_utils import resize_bimask


__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


stemmer = nltk.PorterStemmer()

PAD_TOKEN = ' '
PAD_IDX = 0

UNKNOWN_TOKEN = 'ukn'
UNKNOWN_IDX = 1

MAX_SEQ_LEN = 100
MAX_CHAR_LEN = 10

CHAR_PAD_TOKEN = ' '
CHAR_PAD_IDX = 0

CHAR_UNKNOWN_TOKEN = '~'
CHAR_UNKNOWN_IDX = 1


def dilate_rle(rle, c=1.2, iterations=1):
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.binary_dilation.html
    # https://en.wikipedia.org/wiki/Dilation_%28morphology%29

    highres_bimask = mask_utils.decode(rle)
    h, w = highres_bimask.shape

    E = np.ones((5, 5))

    # Resize to low res
    max_len = 800
    bimask = resize_bimask(highres_bimask, max_len=max_len)

    #     cur_count = np.sum(bimask)
    #     req_count = min(c * cur_count, bimask.size)
    #     new_bimask = np.zeros_like(bimask)
    #     E = np.ones((5, 5))
    #     for i in range(100):
    #         ndimage.morphology.binary_dilation(bimask, output=new_bimask, structure=E)
    #         if np.sum(new_bimask) >= req_count:
    #             break
    #     new_bimask = new_bimask.astype('uint8')

    new_bimask = ndimage.morphology.binary_dilation(bimask, iterations=iterations).astype('uint8')

    # Resize to org size
    new_bimask = imresize(new_bimask, (h, w))
    new_rle = mask_utils.encode(np.asfortranarray(new_bimask))

    del highres_bimask
    del bimask
    del new_bimask

    return new_rle


def seq_to_preds(rows, seq_x, pred_probs, idx_to_attr_id, dilate=False):
    '''
    rows: List of dicts, N elements
    seq_x: Tokens, N x L matrix
    pred_probs: Predicted idxs, N x L x C matrix
    '''

    pred_labels = np.argmax(pred_probs, axis=2)
    # Set as -1 when x_i as PAD_IDX
    pad_idxs = np.where(seq_x == PAD_IDX)
    pred_labels[pad_idxs] = -1

    predictions = []

    n_rows = seq_x.shape[0]
    n_attr = len(idx_to_attr_id)

    progbar = Progbar(n_rows)

    for row_idx, row in enumerate(rows):
        row = rows[row_idx]

        gt_attr_ids = row['attr_id_seq']
        rle_seq = row['rle_seq']
        image_id = row['image_id']

        pred_all_attr_idxs = pred_labels[row_idx]  # A sequence of size L, predictions of PAD_IDXs
        pred_seq_idxs = np.where(pred_all_attr_idxs >= 0)[0]
        pred_attr_idxs = pred_all_attr_idxs[pred_seq_idxs]

        pred_attr_ids = [idx_to_attr_id[a] for a in pred_attr_idxs]

        assert len(pred_attr_ids) == len(gt_attr_ids)

        for w_idx, pred_seq_idx in enumerate(pred_seq_idxs):
            for attr_idx in range(1, n_attr):  # Exclude SAFE
                this_attr_id = idx_to_attr_id[attr_idx]
                this_score = pred_probs[row_idx, pred_seq_idx, attr_idx]
                w_rle = rle_seq[w_idx]

                if this_score > 1e-5 and w_rle is not None:

                    if this_score > 0.05 and dilate:
                        predictions.append({
                            'image_id': image_id,
                            'attr_id': this_attr_id,
                            'segmentation': dilate_rle(w_rle),
                            'score': float(this_score),
                        })
                    else:
                        predictions.append({
                            'image_id': image_id,
                            'attr_id': this_attr_id,
                            'segmentation': w_rle,
                            'score': float(this_score),
                        })
        progbar.update(row_idx)

    return predictions


def compute_accuracy_seq(val_seq_x, val_seq_y, val_pred_labels, attr_ids, attr_id_to_name):
    n_attr = len(attr_ids)
    # Per-class accuracy
    accs = np.zeros((n_attr,))

    # Shape: N x L
    gt_labels = np.argmax(val_seq_y, axis=2)
    pred_labels = np.argmax(val_pred_labels, axis=2)

    # A y_i = 0 could either imply a) x_i = PAD_IDX or b) f(x_i) is Safe
    # So set y_i = -1 when x_i = PAD_IDX
    pad_idxs = np.where(val_seq_x == PAD_IDX)
    gt_labels[pad_idxs] = -1
    pred_labels[pad_idxs] = -1

    for idx in range(n_attr):
        this_attr_gt_labels = gt_labels[gt_labels == idx]
        this_attr_pred_labels = pred_labels[gt_labels == idx]
        accs[idx] = np.sum(this_attr_gt_labels == this_attr_pred_labels) / float(len(this_attr_gt_labels))

    attr_id_acc = zip(attr_ids, accs)

    for attr_id, attr_acc in sorted(attr_id_acc, key=lambda x: -x[1]):
        print '{:>20s}: {:.3f}'.format(attr_id_to_name[attr_id], attr_acc)

    print 'C-Acc = ', np.mean(accs)


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

        feat_mat[start_idx:end_idx] = batch_feat.reshape((this_batch_size, n_feat))
        pbar.update(b_idx)

    # Create a dict: image_id -> feat
    image_id_to_visfeat = dict()
    for i, (image_id, image_path) in enumerate(image_list):
        image_id_to_visfeat[image_id] = feat_mat[i]

    return image_id_to_visfeat


def to_rows(anno, max_seq_len=100, overlap_thresh=0.0):
    rows = []
    n_reassigned = 0

    for image_id, anno in anno.iteritems():
        sequence = anno['sequence']

        if len(sequence['word_attr_ids']) < 1:
            continue

        this_seq_len = len(sequence['word_attr_ids'])

        # Re-assign labels (when overlap is below threshold)
        for i in range(this_seq_len):
            score = sequence['word_overlap_score'][i]
            if score is not None and score < overlap_thresh:  # score is None in case of line breaks
                n_reassigned += 1
                sequence['word_attr_ids'][i] = SAFE_ATTR

        n_batches = (this_seq_len / max_seq_len) + 1
        for batch_idx in range(n_batches):
            start_idx = batch_idx * max_seq_len
            end_idx = min(this_seq_len, start_idx + max_seq_len)

            rows.append({
                'image_id': image_id,
                'text_seq': sequence['word_text'][start_idx:end_idx],
                'attr_id_seq': sequence['word_attr_ids'][start_idx:end_idx],
                'rle_seq': sequence['word_rle'][start_idx:end_idx],
                'poly_seq': sequence['word_poly'][start_idx:end_idx]
            })

    print 'Converted {} labels to SAFE'.format(n_reassigned)
    return rows


def process_word(org_word):
    new_word = org_word.strip().lower()

    # 1. Convert numbers to tokens (replace each digit with a another token (3 in this case)
    new_word = re.sub('\d', '0', new_word)

    #     # 2. If word can be either name or location, use <nameloc>
    #     if new_word in name_and_loc:
    #         new_word = '<nameloc>'

    #     # 3. Replaces names with <name>
    #     if new_word in names_set:
    #         new_word = '<name>'

    #     # 4. Replace locations with <loc>
    #     if new_word in location_set:
    #         new_word = '<loc>'

    # 5. Convert to ascii charset
    if isinstance(new_word, unicode):
        new_word = unicodedata.normalize('NFKD', new_word).encode('ascii', 'ignore')

    # 6. Stem word
    new_word = stemmer.stem(new_word)

    return new_word


def sanitize(org_word):
    new_word = org_word.strip().lower()

    # 1. Convert to ascii charset
    if isinstance(new_word, unicode):
        new_word = unicodedata.normalize('NFKD', new_word).encode('ascii', 'ignore')

    # 2. Stem word
    # new_word = stemmer.stem(new_word)

    return new_word


def to_prior_label(org_word):
    new_word = org_word.strip().lower()

    # 1. Convert numbers to tokens (replace each digit with a another token (3 in this case)
    if any(char.isdigit() for char in new_word):
        new_word = re.sub('\d', '0', new_word)

    # 2. If word can be either name or location, use <nameloc>
    elif new_word in name_and_loc:
        new_word = 'victoria'

    # 3. Replaces names with <name>
    elif new_word in names_set:
        new_word = 'john'

    # 4. Replace locations with <loc>
    elif new_word in location_set:
        new_word = 'berlin'

    else:
        new_word = '<ig>'  # ignore

    # 5. Convert to ascii charset
    if isinstance(new_word, unicode):
        new_word = unicodedata.normalize('NFKD', new_word).encode('ascii', 'ignore')

    return new_word


def process_rows(rows):
    n_rows = len(rows)
    pbar = Progbar(n_rows)

    for row_idx, row in enumerate(rows):
        text_seq = row['text_seq']

        row['org_text_seq'] = map(sanitize, text_seq)
        row['prior_seq'] = map(to_prior_label, text_seq)
        row['proc_text_seq'] = map(process_word, text_seq)

        pbar.update(row_idx)

    return rows


def rows_to_word_feature_labels(rows, word_to_idx, attr_id_to_idx):
    word_idxs = []  # For each word in the dataset, add the corresponding word_idx
    label_idxs = []
    for row in rows:
        for word, attr_id in zip(row['text_seq'], row['attr_id_seq']):
            # Word
            this_word_idx = word_to_idx.get(word.lower(), UNKNOWN_IDX)
            word_idxs.append(this_word_idx)

            # Label
            this_attr_idx = attr_id_to_idx[attr_id]
            label_idxs.append(this_attr_idx)

    features = np.asarray(word_idxs)[:, None]
    labels = keras.utils.to_categorical(label_idxs, num_classes=len(attr_id_to_idx))

    return features, labels


def rows_to_seq_feature_labels(rows, img_id_to_visfeat, word_to_idx, char_to_idx, attr_id_to_idx):
    n_rows = len(rows)
    max_seq_len = MAX_SEQ_LEN
    max_char_len = MAX_CHAR_LEN
    n_attr = len(attr_id_to_idx)

    SAFE_ATTR_IDX = attr_id_to_idx[SAFE_ATTR]

    word_idx_seqs = []
    proc_word_idx_seqs = []
    prior_idx_seqs = []
    labels_seqs = []
    # N x L x K
    # char_idx_seqs = np.ones((n_rows, max_seq_len, max_char_len)) * CHAR_PAD_IDX
    char_idx_seqs = []

    # FIXME
    visfeat_dim = img_id_to_visfeat.values()[0].shape[0]
    # visfeat_dim = 10
    img_feat = np.zeros((n_rows, visfeat_dim))

    for row_idx, row in enumerate(rows):
        # --- Word
        # Convert list of words into list of word_idxs
        word_list = row['{}text_seq'.format('org_')]
        word_idx_list = [word_to_idx.get(w.lower(), UNKNOWN_IDX) for w in word_list]
        word_idx_seqs.append(word_idx_list)

        # --- Pre-processed Word
        # Convert list of words into list of word_idxs
        word_list = row['{}text_seq'.format('proc_')]
        word_idx_list = [word_to_idx.get(w.lower(), UNKNOWN_IDX) for w in word_list]
        proc_word_idx_seqs.append(word_idx_list)

        # --- Priors
        # Convert list of words into list of word_idxs
        word_list = row['prior_seq']
        word_idx_list = [word_to_idx.get(w.lower(), UNKNOWN_IDX) for w in word_list]
        prior_idx_seqs.append(word_idx_list)

        # -- Label
        attr_id_list = row['attr_id_seq']
        attr_idx_list = [attr_id_to_idx.get(a, SAFE_ATTR_IDX) for a in attr_id_list]
        labels_seqs.append(attr_idx_list)

        # --- Image
        image_id = row['image_id']
        # FIXME
        img_feat[row_idx] = img_id_to_visfeat[image_id]

        # --- Characters
        char_idx_list = []
        org_word_list = row['text_seq']
        for word in org_word_list:
            char_idx_list.append([char_to_idx.get(_c, CHAR_UNKNOWN_IDX) for _c in word.lower()])
        word_chars = pad_sequences(char_idx_list, maxlen=max_char_len)  # W x K
        char_idx_seqs.append(word_chars)

    # N = n_rows,  L = max_seq_len,   C = # classes
    word_idx_seqs = pad_sequences(word_idx_seqs, maxlen=max_seq_len)  # N x L
    proc_word_idx_seqs = pad_sequences(proc_word_idx_seqs, maxlen=max_seq_len)  # N x L
    prior_idx_seqs = pad_sequences(prior_idx_seqs, maxlen=max_seq_len)  # N x L
    labels_seqs = pad_sequences(labels_seqs, maxlen=max_seq_len)  # N x L
    char_idx_seqs = pad_sequences(char_idx_seqs, maxlen=max_seq_len)  # N x L x K

    # Convert label seqs to categorical
    labels_mat = np.zeros((n_rows, max_seq_len, n_attr))
    for i in range(n_rows):
        labels_mat[i, :, :] = keras.utils.to_categorical(labels_seqs[i], num_classes=n_attr)

    # Normalize image features
    img_feat = img_feat / np.linalg.norm(img_feat, axis=1)[:, None]

    return word_idx_seqs, proc_word_idx_seqs, prior_idx_seqs, labels_mat, img_feat, char_idx_seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_proxy", type=str, help="Path to Train proxy file")
    parser.add_argument("test_proxy", type=str, help="Path to Test proxy file")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    parser.add_argument("-d", "--device_id", type=str, help="GPU ID", default='0')
    args = parser.parse_args()
    params = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = params['device_id']

    # --- Load some necessary helpers ----------------------------------------------------------------------------------
    image_index = get_image_id_info_index()
    attr_id_to_name = load_attributes_shorthand()

    # --- Load data ----------------------------------------------------------------------------------------------------
    train_anno = json.load(open(params['train_proxy']))
    val_anno = json.load(open(params['test_proxy']))

    print '# Train images = ', len(train_anno)
    print '# Test images = ', len(val_anno)

    attr_ids = [SAFE_ATTR, ] + MODE_TO_ATTR_ID['textual']
    n_attr = len(attr_ids)

    attr_id_to_idx = dict(zip(attr_ids, range(n_attr)))
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}
    print '# Attributes = ', n_attr
    print 'Attributes: '
    print attr_ids

    SAFE_ATTR_IDX = attr_id_to_idx[SAFE_ATTR]

    # --- Extract Image Features ---------------------------------------------------------------------------------------
    train_image_list = []  # (image_id, image_path)
    val_image_list = []  # (image_id, image_path)

    for image_id, entry in train_anno.iteritems():
        train_image_list.append((image_id, entry['image_path']))

    for image_id, entry in val_anno.iteritems():
        val_image_list.append((image_id, entry['image_path']))

    train_imgarr = image_list_to_arr(train_image_list)
    print
    val_imgarr = image_list_to_arr(val_image_list)

    print
    print train_imgarr.shape
    print val_imgarr.shape

    base_resnet = ResNet50(include_top=False, weights='imagenet')
    resnet = Model(inputs=base_resnet.input, outputs=base_resnet.get_layer('avg_pool').output)

    x_train_img_id_to_visfeat = img_to_features(train_imgarr, train_image_list, resnet)
    print
    x_val_img_id_to_visfeat = img_to_features(val_imgarr, val_image_list, resnet)

    print
    print x_train_img_id_to_visfeat.values()[0].shape
    print x_val_img_id_to_visfeat.values()[0].shape

    # --- Convert sequences to required format -------------------------------------------------------------------------
    train_rows = to_rows(train_anno)
    val_rows = to_rows(val_anno)
    print

    print '# train_rows = ', len(train_rows)
    print '# val_rows = ', len(val_rows)

    # --- Preprocess/Sanitize text -------------------------------------------------------------------------------------
    # ----------- Names
    fnames_path = '/BS/orekondy2/work/privacy_filters/cache/names/fnames.txt'
    lnames_path = '/BS/orekondy2/work/privacy_filters/cache/names/lnames.txt'

    global names_set
    names_set = set()

    for name_path in [fnames_path, lnames_path]:
        for _name in open(name_path):
            names_set.add(_name.lower().strip())

    print 'Loaded {} names...'.format(len(names_set))

    # ----------- Locations
    loc_path = '/BS/orekondy2/work/privacy_filters/cache/locations/cities_and_countries.txt'
    global location_set
    location_set = set()

    # For more information: http://download.geonames.org/export/dump/
    with open(loc_path) as rf:
        for line in rf:
            loc = line.strip()
            location_set.add(loc.lower())

    print 'Loaded {} locations'.format(len(location_set))

    # ----------- Names & Locations
    global name_and_loc
    name_and_loc = names_set & location_set
    print 'Name & Loc = ', len(name_and_loc)

    # ----------- Process now
    train_rows = process_rows(train_rows)
    print
    val_rows = process_rows(val_rows)
    print

    print '# train_rows = ', len(train_rows)
    print '# val_rows = ', len(val_rows)
    print

    # --- Tokenize Words -----------------------------------------------------------------------------------------------
    use_processed_words = True
    embed_priors = False
    textproc = 'proc_' if use_processed_words else 'org_'

    word_counter = Counter()

    for row in train_rows:
        for word in row['{}text_seq'.format(textproc)]:
            word_counter[word.lower()] += 1
        if embed_priors:
            for word in row['prior_seq']:
                word_counter[word.lower()] += 1

    print 'Most common words: '
    print word_counter.most_common(n=20)

    print 'Vocab size = ', len(word_counter)
    print '# >= 2 occurences = ', len(filter(lambda x: x >= 2, word_counter.values()))
    print '# >= 3 occurences = ', len(filter(lambda x: x >= 3, word_counter.values()))
    print '# >= 4 occurences = ', len(filter(lambda x: x >= 4, word_counter.values()))

    # ** Tune this **
    # VOCAB_SIZE = 7108   # For original
    # VOCAB_SIZE = 8768  # Processed
    VOCAB_SIZE = 4137
    print 'Setting VOCAB_SIZE = ', VOCAB_SIZE

    # Create dict: word -> idx
    WORD_TO_IDX = dict()
    WORD_TO_IDX[PAD_TOKEN] = PAD_IDX
    WORD_TO_IDX[UNKNOWN_TOKEN] = UNKNOWN_IDX

    next_idx = max(PAD_IDX, UNKNOWN_IDX) + 1

    for idx, (word, word_count) in enumerate(word_counter.most_common()):
        if idx >= VOCAB_SIZE - 2:
            break
        else:
            WORD_TO_IDX[word] = next_idx
            next_idx += 1

    print 'Vocab look-up dict size = ', len(WORD_TO_IDX)

    KNOWN_WORDS = set(WORD_TO_IDX.keys())

    # ----------- Load Embedding
    EMBEDDING_DIM = 100
    GLOVE_DIR = '/BS/orekondy2/work/privacy_filters/cache/embeddings'
    KNOWN_WORDS = set(WORD_TO_IDX.keys())

    embedding_filename = 'glove.6B.100d.txt'
    # embedding_filename = 'glove.twitter.27B.100d.txt'
    # embedding_filename = 'glove.840B.300d.txt'

    # Get embedding matrix
    embeddings_index = dict()
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    words_found = set()

    with open(osp.join(GLOVE_DIR, embedding_filename)) as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in KNOWN_WORDS:
                words_found.add(word)
                word_idx = WORD_TO_IDX[word]
                embedding_matrix[word_idx] = np.asarray(values[1:], dtype='float32')

    print '# Words found in pre-loaded embeddings = {} / {}'.format(len(words_found), VOCAB_SIZE)

    print 'Most common words not found in embed dictionary: '
    for word, wc in filter(lambda x: x[0] not in words_found, word_counter.most_common())[:20]:
        print word, wc

    # --- Tokenize Characters ------------------------------------------------------------------------------------------
    char_counter = Counter()

    for row in train_rows:
        for word in row['text_seq']:
            for _c in word.lower():
                char_counter[_c] += 1

    print 'Most common characters: '
    print char_counter.most_common(n=10)

    print 'Vocab size = ', len(char_counter)
    print '# >= 2 occurences = ', len(filter(lambda x: x >= 2, char_counter.values()))
    print '# >= 3 occurences = ', len(filter(lambda x: x >= 3, char_counter.values()))
    print '# >= 4 occurences = ', len(filter(lambda x: x >= 4, char_counter.values()))

    CHAR_VOCAB_SIZE = 110

    # Create dict: char -> idx
    CHAR_TO_IDX = dict()
    CHAR_TO_IDX[PAD_TOKEN] = CHAR_PAD_IDX
    CHAR_TO_IDX[UNKNOWN_TOKEN] = CHAR_UNKNOWN_IDX

    next_idx = max(CHAR_PAD_IDX, CHAR_UNKNOWN_IDX) + 1

    for idx, (_c, char_count) in enumerate(char_counter.most_common()):
        if idx >= CHAR_VOCAB_SIZE - 2:
            break
        else:
            CHAR_TO_IDX[_c] = next_idx
            next_idx += 1

    print 'Vocab look-up dict size = ', len(CHAR_TO_IDX)

    KNOWN_CHARS = set(CHAR_TO_IDX.keys())

    # --- Create Training Data -----------------------------------------------------------------------------------------
    # x_train_img_id_to_visfeat = dict()
    # x_val_img_id_to_visfeat = dict()
    train_seq_x, train_procseq_x, train_seq_x_prior, train_seq_y, train_img_x, train_charseq_x = rows_to_seq_feature_labels(
        train_rows, x_train_img_id_to_visfeat, WORD_TO_IDX, CHAR_TO_IDX, attr_id_to_idx)
    val_seq_x, val_procseq_x, val_seq_x_prior, val_seq_y, val_img_x, val_charseq_x = rows_to_seq_feature_labels(
        val_rows, x_val_img_id_to_visfeat, WORD_TO_IDX, CHAR_TO_IDX, attr_id_to_idx)

    print train_seq_x.shape, train_procseq_x.shape, train_seq_x_prior.shape, train_seq_y.shape, train_img_x.shape, train_charseq_x.shape
    print val_seq_x.shape, val_procseq_x.shape, val_seq_x_prior.shape, val_seq_y.shape, val_img_x.shape, val_charseq_x.shape

    # --- Model -----------------------------------------------------------------------------------------
    K.clear_session()
    # ------ Define Model
    vis_feat_length = train_img_x[0].shape[0]
    xin = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    xin_img = Input(shape=(vis_feat_length,))

    # Embed
    embed_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix])
    embedding = embed_layer(xin)

    # Sequence
    text_seq = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(
        embedding)
    text_seq = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(
        text_seq)

    # Compress Image Features
    vis_feat = Dense(256, activation='relu')(xin_img)
    vis_feat = Dropout(0.2)(vis_feat)
    # Repeat image feature for each time step
    vis_seq = keras.layers.RepeatVector(MAX_SEQ_LEN)(vis_feat)
    # Merge this at each time-step
    merged_seq = keras.layers.multiply([text_seq, vis_seq])

    # MLP
    mlp = TimeDistributed(Dense(n_attr, activation='softmax'))(merged_seq)

    model = Model(inputs=[xin, xin_img], outputs=mlp)
    # model.compile(optimizer='Adam', loss='categorical_crossentropy')
    model.summary()

    # ------ Train
    batch_size = 128
    epochs = 50

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=[metrics.categorical_accuracy])

    history = model.fit([train_procseq_x, train_img_x], train_seq_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)

    # ------ Predict
    val_pred_labels = model.predict([val_procseq_x, val_img_x], batch_size=batch_size, verbose=1)
    print
    compute_accuracy_seq(val_procseq_x, val_seq_y, val_pred_labels, attr_ids, attr_id_to_name)

    # --- Write output
    out_path = params['outfile']
    predictions = seq_to_preds(val_rows, val_seq_x, val_pred_labels, idx_to_attr_id, dilate=False)

    print
    print 'Writing {} predictions to {}'.format(len(predictions), out_path)
    json.dump(predictions, open(out_path, 'wb'), indent=2)



if __name__ == '__main__':
    main()