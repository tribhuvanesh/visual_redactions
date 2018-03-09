#!/usr/bin/python
"""Given an annotation file, produce visualization of labels.

Given an annotation file, produce these visualization images (one per attribute present in the image):
  a. segmentation mask
  b. instance mask
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

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def create_individual_visuals(anno_filepath, output_dir):
    with open(anno_filepath) as jf:
        _annos = json.load(jf)
    annos = _annos['annotations']

    for file_id, entry in annos:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_filepath", type=str, help="Path to Annotation file")
    parser.add_argument("output_dir", type=str, help="Directory to place output images")
    args = parser.parse_args()

    create_individual_visuals(args.annotation_filepath, args.output_dir)


if __name__ == '__main__':
    main()