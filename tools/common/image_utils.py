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

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def resize_min_side(pil_img, mins_len):
    """
    Scale image such that minimum side is length mins_len pixels
    :param pil_img: PIL image
    :param mins_len: Size of minimum side after rescaling
    :return:
    """
    # What's the min side?
    w, h = pil_img.size
    if w < h:
        new_w = mins_len
        new_h = int(np.round(h * (new_w / float(w))))   # Scale height to same aspect ratio
    else:
        new_h = mins_len
        new_w = int(np.round(w * (new_h / float(h))))   # Scale height to same aspect ratio
    return pil_img.resize((new_w, new_h))