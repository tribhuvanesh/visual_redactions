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

from tqdm import tqdm

from privacy_filters.config import *
from privacy_filters.tools.common.utils import *

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags

from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger(osp.join(Paths.CACHE_PATH, 'stanford-ner-2014-06-16', 'classifiers',
                                'english.muc.7class.caseless.distsim.crf.ser.gz'),
                       '/BS/orekondy2/work/opt/CoreNLP/stanford-corenlp.jar',
                    encoding='utf-8')


__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_proxy", type=str, help="Path to Test proxy file")
    parser.add_argument("outfile", type=str, help="Path to write predictions")
    args = parser.parse_args()
    params = vars(args)

    # --- Load some necessary helpers ----------------------------------------------------------------------------------
    image_index = get_image_id_info_index()
    attr_id_to_name = load_attributes_shorthand()

    # --- Load data ----------------------------------------------------------------------------------------------------
    val_anno = json.load(open(params['test_proxy']))

    print '# Test images = ', len(val_anno)

    attr_ids = [SAFE_ATTR, ] + MODE_TO_ATTR_ID['textual']
    n_attr = len(attr_ids)

    attr_id_to_idx = dict(zip(attr_ids, range(n_attr)))
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}
    print '# Attributes = ', n_attr

    # --- Entity -> Attribute ------------------------------------------------------------------------------------------
    entity_to_attr_ids = {
        'PERSON': [
            "a111_name_all",
        ],

        'ORGANIZATION': ["a111_name_all",
                         ],

        'DATE': ["a82_date_time",
                 "a24_birth_date", ],

        'LOCATION': ["a106_address_current_all",
                     "a107_address_home_all",
                     "a73_landmark", ],

        'TIME': ["a82_date_time", ],

        'MONEY': [],
        'PERCENT': [],
        'O': []
    }

    # --- NER ----------------------------------------------------------------------------------------------------------
    predictions = []

    n_imgs = len(val_anno)
    with tqdm(total=n_imgs) as pbar:
        for idx, (image_id, entry) in enumerate(val_anno.iteritems()):
            sequence = entry['sequence']
            word_seq = sequence['word_text']

            # Manually tokenize sequence
            new_word_seq = []
            new_word_rle = []

            for word, word_rle in zip(sequence['word_text'], sequence['word_rle']):
                if word.startswith('<') and word.endswith('>'):
                    pass
                else:
                    new_word_seq.append(word.lower())
                    new_word_rle.append(word_rle)

            # NER
            tags = st.tag(new_word_seq)
            tags = [x[1] for x in tags]

            # Iterate through image-level tags. If word is tagged, create a prediction.
            for etype, word_rle, word_text in zip(tags, new_word_rle, new_word_seq):
                for attr_id in entity_to_attr_ids[etype]:
                    predictions.append({
                        'image_id': image_id,
                        'attr_id': attr_id,
                        'segmentation': word_rle,
                        'score': 1.0,
                        'text': word_text
                    })
            pbar.update()

    # --- Write Output -------------------------------------------------------------------------------------------------
    outfile = params['outfile']
    print 'Writing {} predictions to {}'.format(len(predictions), outfile)
    with open(outfile, 'w') as wf:
        json.dump(predictions, wf, indent=2)


if __name__ == '__main__':
    main()