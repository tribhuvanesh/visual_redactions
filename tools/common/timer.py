#!/usr/bin/python
"""A Timer to ease profiling.

Source: https://www.huyng.com/posts/python-performance-analysis
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
import time

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs