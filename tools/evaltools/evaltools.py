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

from PIL import Image, ImageDraw
from scipy.misc import imread
from matplotlib.path import Path as mpath

from privacy_filters.tools.common.plot_utils import gen_n_colors

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def get_mask(nx, ny, poly_verts_list, retain_instances=False, return_grid_list=False):
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

    if return_grid_list:
        return grid_list

    if not retain_instances:
        combined_grid = reduce(lambda x, y: x | y, grid_list)
    else:
        combined_grid = np.zeros_like(grid_list[0])
        for idx in range(len(grid_list)):
            # Tag each instance with a particular number
            instance_id = idx + 1
            this_instance_grid = grid_list[idx]
            # Find pixels that have not been tagged with an instance yet
            unselected_pixels = np.logical_and(combined_grid == 0, this_instance_grid == 1)
            combined_grid[unselected_pixels] = instance_id
    return combined_grid.astype(int)


def via_regions_to_polygons(via_regions, include_instance=False):
    """

    :param via_regions:
    :param include_instance:
    :return: A list of polygons [ [(x1, y1), (x2, y2), ...], ]
    """
    poly_verts_list = []
    instance_list = []
    for idx, (region_id, region_dct) in enumerate(via_regions.iteritems()):
        try:
            poly_verts = zip(region_dct['shape_attributes']['all_points_x'], region_dct['shape_attributes']['all_points_y'])
        except KeyError:
            print "region_dct['shape_attributes'].keys() = ", region_dct['shape_attributes'].keys()
            print region_dct['shape_attributes']
            raise
        poly_verts_list.append(poly_verts)
        if 'assigned_instance_id' in region_dct:
            instance_list.append(region_dct['assigned_instance_id'])

    if include_instance:
        return poly_verts_list, instance_list
    else:
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


def visualize_masks(im, mask_list, img_out_path):
    """

    :param im: PIL Image
    :param mask: mask containing ints \in [-1, 0, 1, .., N]
    :return:
    """
    plt.clf()
    plt.figure(figsize=(15, 10))
    w, h = im.size

    # Actual Image
    plt.subplot(211)
    plt.axis('off')
    plt.imshow(im)
    plt.title('Image')

    # Overlay
    plt.subplot(212)
    plt.axis('off')
    plt.imshow(im)
    # Sort masks by size of this instance
    mask_list = sorted(mask_list, key=lambda x: np.sum(x))
    colors = gen_n_colors(len(mask_list))
    for i, mask in enumerate(mask_list):
        colored_mask_im = convert_mask_to_img(mask, hot_color=colors[i])
        plt.imshow(colored_mask_im, alpha=0.6)
    plt.title('Instances')

    plt.savefig(img_out_path)
    plt.close('all')


def visualize_polygons(im, poly_verts_list, img_out_path, instances=None):
    """

    :param im:
    :param poly_verts_list:
    :param img_out_path:
    :param instances:
    :return:
    """
    plt.clf()
    plt.figure(figsize=(15, 10))
    w, h = im.size

    # Actual Image
    plt.subplot(211)
    plt.axis('off')
    plt.imshow(im)
    plt.title('Image')

    # Overlay
    plt.subplot(212)
    plt.axis('off')
    if instances is None:
        colors = gen_n_colors(len(poly_verts_list) + 1)
    else:
        colors = gen_n_colors(len(set(instances)) + 1)
    # Draw polygons on image
    imp = im.copy()
    if imp.mode not in ('RGB', 'RGBA'):
        imp = imp.convert(mode='RGB')
    draw = ImageDraw.Draw(imp, 'RGBA')
    for idx, poly in enumerate(poly_verts_list):
        if instances is None:
            fill_col = (int(colors[idx][0]), int(colors[idx][1]), int(colors[idx][2]), 200)
        else:
            i_id = instances[idx]
            # print 'i_id = ', i_id
            # print 'len(colors) = ', len(colors)
            fill_col = (int(colors[i_id][0]), int(colors[i_id][1]), int(colors[i_id][2]), 200)
        draw.polygon(poly_verts_list[idx], fill=fill_col, outline=(0, 0, 0, 255))
    del draw
    plt.imshow(imp)
    plt.title('Instances')

    plt.savefig(img_out_path)
    plt.close('all')


def visualize_errors(im, gt_mask, pred_mask, img_out_path, metrics_text=''):
    """
    Masks contain [-1, 0, 1]
    -1 indicates ignore regions
    """
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
