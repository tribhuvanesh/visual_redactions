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
from scipy import ndimage

import pycocotools.mask as mask

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class ImageAnnotation:
    def __init__(self, image_id, image_path, image_height, image_width):
        self.image_id = image_id
        self.image_path = image_path
        self.image_height = image_height
        self.image_width = image_width
        self.attributes = []
        self.next_instance_id = 0

    def validate_attr_annotation(self, attr_anno):
        for polygon in attr_anno.polygons:
            # Check bounds for x and y
            all_x = [polygon[i] for i in range(len(polygon)) if i % 2 == 0]
            all_y = [polygon[i] for i in range(len(polygon)) if i % 2 != 0]
            assert [x <= self.image_width for x in all_x], 'Found values: {} above image width (={})'.format(
                filter(lambda w: w > self.image_width, all_x), self.image_width)
            assert [y <= self.image_height for y in all_y], 'Found values: {} above image height (={})'.format(
                filter(lambda w: w > self.image_height, all_y), self.image_height)

    def add_attribute_annotation(self, attr_anno):
        self.validate_attr_annotation(attr_anno)
        attr_anno.set_instance_id(self.next_instance_id)
        self.next_instance_id += 1
        self.attributes.append(attr_anno)

    def finalize(self):
        """
        Infer additional image-level attributes
        :return:
        """
        # Finalize attributes
        for attr in self.attributes:
            attr.finalize(self.image_height, self.image_width)
        # Clean-up image path (make this relative to DS_ROOT)
        self.image_path = self.image_path[self.image_path.index('images'):]


class AttributeAnnotation:
    """
    Represents annotation of a single unique instance of attribute attr_id
    """
    def __init__(self, instance_id, attr_id, polygons, is_crowd=False):
        self.instance_id = instance_id
        self.attr_id = attr_id
        self.polygons = []
        for poly in polygons:
            self.add_polygon(poly)
        self.area = []            # One number per region
        self.segmentation = None  # COCO RLE
        self.bbox = []            # Stored as [ [x y w h], ... ]
        self.iscrowd = False

    def set_instance_id(self, new_instance_id):
        self.instance_id = new_instance_id

    def add_polygon(self, polygon):
        assert all([type(z) in [int, float, np.float, np.float64] for z in polygon])  # Contains only numbers
        assert all([z >= 0. for z in polygon]), 'Found negative values in polygon: {}'.format(
            filter(lambda z: z < 0., polygon))
        assert len(polygon) >= 6  # Contains at least 3 points
        assert polygon[0] == polygon[-2]  # x1 == xn
        assert polygon[1] == polygon[-1]  # y1 == yn
        self.polygons.append(polygon)

    def finalize(self, image_height, image_width):
        """
        Infer additional details per attribute (eg., area of instance, RLE encoding of polygon, etc.)
        :return:
        """
        rles = mask.frPyObjects(self.polygons, image_height, image_width)
        rle = mask.merge(rles)
        self.segmentation = rle
        self.area = mask.area(rle).tolist()
        self.bbox = mask.toBbox(rle).tolist()


class AnnoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AttributeAnnotation):
            return obj.__dict__
        elif isinstance(obj, ImageAnnotation):
            dct = obj.__dict__
            del(dct['next_instance_id'])   # Because this is unnecessary now
            return dct
        return json.JSONEncoder.default(self, obj)


def dilate_rle(rle, c):
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.binary_dilation.html
    # https://en.wikipedia.org/wiki/Dilation_%28morphology%29

    bimask = mask.decode(rle)
    new_bimask = np.zeros_like(bimask)

    cur_count = np.sum(bimask)
    req_count = min(c * cur_count, bimask.size)

    # Superimpose this structure at every 1 in the binary mask
    # E = np.ones((5, 5))
    max_iter = 100

    for _i in range(max_iter):
        ndimage.morphology.binary_dilation(bimask, output=new_bimask)
        if np.sum(new_bimask) > req_count:
            break

    new_rle = mask.encode(new_bimask)

    del bimask
    del new_bimask

    return new_rle