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
import colorsys

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def gen_n_colors(n):
    hsv_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    rgb_tuples = map(lambda y: map(lambda x: int(255*x), y), rgb_tuples)
    return rgb_tuples
