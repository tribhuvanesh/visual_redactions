#!/usr/bin/python
"""Summarizes a summary file (to paste into spreadsheet)

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

from privacy_filters.tools.evaltools.evaluate_simple import MODE_TO_ATTR_ID
from privacy_filters.tools.common.utils import load_attributes_shorthand

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_path", type=str, help="Path to summary json file")
    args = parser.parse_args()

    with open(args.summary_path) as jf:
        summ = json.load(jf)

    attr_id_to_name = load_attributes_shorthand()

    print '--- Overall ---'
    print summ['overall']['iou']
    print

    for _mode, attr_ids in MODE_TO_ATTR_ID.iteritems():
        print '--- Mode = ', _mode, '---'
        attr_names = [attr_id_to_name[attr_id] for attr_id in attr_ids]
        print '\t'.join(attr_names)
        ious = [summ['per_attribute'][attr_id]['iou'] for attr_id in attr_ids]
        print '\t'.join(map(str, ious))
        print 'mIoU = ', np.mean(ious)
        print


if __name__ == '__main__':
    main()