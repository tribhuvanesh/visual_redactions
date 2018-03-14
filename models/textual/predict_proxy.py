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

from privacy_filters.config import *

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Path to proxy file")
    parser.add_argument("outfile", type=str, help="Output path")
    args = parser.parse_args()
    params = vars(args)

    infile = params['infile']
    outfile = params['outfile']

    anno = json.load(open(infile, 'rb'))
    anno = dict(filter(lambda x: 'sequence' in x[1], anno.iteritems()))

    predictions = []

    for image_idx, (image_id, entry) in enumerate(anno.iteritems()):
        sequence = entry['sequence']

        for attr_id, rle, score, text in zip(sequence['word_attr_ids'],
                                             sequence['word_rle'],
                                             sequence['word_overlap_score'],
                                             sequence['word_text']):
            if attr_id is not SAFE_ATTR and rle is not None:
                predictions.append({
                    'image_id': image_id,
                    'attr_id': attr_id,
                    'segmentation': rle,
                    'score': score,
                    'text': text,
                })

    print 'Writing {} predictions to {}'.format(len(predictions), outfile)
    with open(outfile, 'w') as wf:
        json.dump(predictions, wf, indent=2)


if __name__ == '__main__':
    main()