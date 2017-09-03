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