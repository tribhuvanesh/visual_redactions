#!/usr/bin/python
"""Helper functions for annotations retrieved from Google Clous Vision API.

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

from privacy_filters import DS_ROOT, SEG_ROOT

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


EXTRA_ANNO_PATH = osp.join(SEG_ROOT, 'annotations-extra')


def bb_to_verts(bb):
    vrts_dct = bb['vertices']  # List of [ {x:__, y:__}, ... ]
    vrts = [[d.get('x', 0), d.get('y', 0)] for d in vrts_dct]  # Convert to [ [x1, y1], [x2, y2], ..]
    vrts.append(vrts[0])  # Reconnect to first vertex
    return np.ndarray.flatten(np.asarray(vrts))


def load_image_id_to_text():
    index_path = osp.join(SEG_ROOT, 'privacy_filters', 'cache', 'image_id_text_index.pkl')
    if osp.exists(index_path):
        print 'Found cached img-text index. Loading it...'
        return pickle.load(open(index_path, 'rb'))
    else:
        print 'Creating img-text index ...'
        img_text_index = dict()
        for fold in ['train2017', 'val2017', 'test2017', ]:
            fold_dir = osp.join(EXTRA_ANNO_PATH, fold)
            for filename in os.listdir(fold_dir):
                image_id, _ = osp.splitext(filename)
                with open(osp.join(fold_dir, filename)) as jf:
                    eanno = json.load(jf)
                    if 'fullTextAnnotation' in eanno:
                        this_text = eanno['fullTextAnnotation']['text']
                        img_text_index[image_id] = this_text
        pickle.dump(img_text_index, open(index_path, 'wb'))
        return img_text_index