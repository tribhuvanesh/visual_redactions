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
from matplotlib.path import Path as mpath

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def get_mask(nx, ny, poly_verts_list):
    """
    Represent a set of polygons on grid size (nx x ny) as a mask. This mask will be of size (nx, ny), the
    polygons represented 1s and the remaining areas with 0s.

    :param nx: grid width
    :param ny: grid height
    :param poly_verts_list: List of polygons in the format [ [(x1, y1), (x2, y2), ...], ...]
    :return:
    """
    if poly_verts_list == []:
        # In case there are no polygons, simply return an empty canvas
        return np.zeros((ny, nx))
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    # Source: https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    grid_list = []
    for poly_verts in poly_verts_list:
        path = mpath(poly_verts)
        grid = path.contains_points(points)
        grid = grid.reshape((ny, nx))
        grid_list.append(grid)

    combined_grid = reduce(lambda x, y: x | y, grid_list)
    return combined_grid.astype(int)


def via_regions_to_poygons(via_regions):
    poly_verts_list = []
    for idx, (region_id, region_dct) in enumerate(via_regions.iteritems()):
        poly_verts = zip(region_dct['shape_attributes']['all_points_x'], region_dct['shape_attributes']['all_points_y'])
        poly_verts_list.append(poly_verts)
    return poly_verts_list


def resize_polygons(poly_verts_list, x_shrink_factor=1.0, y_shrink_factor=1.0):
    # Polygon format is [ [(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...]
    return map(lambda poly: map(lambda xy: (xy[0] * x_shrink_factor, xy[1] * y_shrink_factor), poly), poly_verts_list)


def compute_eval_metrics(gt_mask, pred_mask):
    """
    Evaluate a mask w.r.t a GT mask
    :param gt_mask: m x n grid of 0s and 1s
    :param pred_mask: m x n grid of 0s and 1s
    :return:
    """
    assert (gt_mask.size == pred_mask.size)

    if np.sum(gt_mask) == 0 or np.sum(pred_mask) == 0:
        # Handle the special case where only a single mask is annotated and the other hasn't been
        prec, rec, iou = 0.0, 0.0, 0.0
    else:
        tp = float(np.sum(np.logical_and(pred_mask == 1, gt_mask == 1)))
        fp = float(np.sum(np.logical_and(pred_mask == 1, gt_mask == 0)))
        fn = float(np.sum(np.logical_and(pred_mask == 0, gt_mask == 1)))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        iou = tp / (tp + fn + fp)

    return prec, rec, iou


def convert_mask_to_img(mask, hot_color):
    assert len(hot_color) == 3

    # Create a canvas filled with hot_color
    h, w = mask.shape
    im = np.zeros((h, w, 3))
    im[:, :] = hot_color
    im[:, :] *= mask[:, :, None]
    return im


def visualize_errors(im, gt_mask, pred_mask, img_out_path, metrics_text=''):
    '''
    Masks contain [-1, 0, 1]
    -1 indicates ignore regions
    '''
    plt.clf()
    plt.figure(figsize=(20, 15))
    w, h = im.size

    assert (gt_mask.size == pred_mask.size)
    tp = np.logical_and(pred_mask == 1, gt_mask == 1)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    tn = np.logical_and(pred_mask == 0, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)

    # Actual Image
    plt.subplot(221)
    plt.axis('off')
    plt.imshow(im)
    plt.title('Image')

    # True Positives
    plt.subplot(222)
    plt.axis('off')
    plt.imshow(im, alpha=0.5)
    plt.imshow(convert_mask_to_img(tp, [0, 1, 0]), alpha=0.8)
    plt.title('True Positives')

    # False Negatives
    plt.subplot(223)
    plt.axis('off')
    plt.imshow(im, alpha=0.5)
    plt.imshow(convert_mask_to_img(fn, [1, 0, 0]), alpha=0.8)
    plt.title('False Negatives')

    # False Positives
    plt.subplot(224)
    plt.axis('off')
    plt.imshow(im, alpha=0.5)
    plt.imshow(convert_mask_to_img(fp, [1, 1, 0]), alpha=0.8)
    plt.title('False Positives')

    plt.suptitle(metrics_text)

    # plt.tight_layout()
    plt.savefig(img_out_path)
    plt.close('all')
