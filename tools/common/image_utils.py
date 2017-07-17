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

from PIL import Image, ImageDraw, ImageFilter
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


def get_image_size(img_path):
    """
    Get image size as (width, height)
    :param img_path:
    :return: (width, height)
    """
    im = Image.open(img_path)
    return im.size


def draw_outline_on_img(pil_img, poly, color='yellow', width=4):
    im = pil_img.copy()
    draw = ImageDraw.Draw(im)
    draw.line(poly, fill=color, width=4)
    del draw
    return im


def blur_region(org_im, poly, radius=2):
    im = org_im.copy()

    # Blur the entire image
    blurred_image = im.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_im_array = np.asarray(blurred_image)

    # Generate a mask for the polygon
    im_array = np.asarray(im).copy()
    mask_im = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
    ImageDraw.Draw(mask_im).polygon(poly, outline=1, fill=1)
    mask = np.array(mask_im)

    # Copy this region from the blurred image on to the original
    im_array[mask.astype(bool)] = blurred_im_array[mask.astype(bool)]

    return Image.fromarray(im_array)


def fill_region(pil_img, poly, color='yellow'):
    im = pil_img.copy()
    draw = ImageDraw.Draw(im)
    draw.polygon(poly, fill=color)
    del draw

    return im


def crop_region(org_im, poly, return_cropped=True, return_grayscale=False, bkg_fill=255):
    im = org_im.copy()

    # Generate a mask for the polygon
    im_array = np.asarray(im).copy()
    mask_im = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
    ImageDraw.Draw(mask_im).polygon(poly, outline=1, fill=1)
    mask = np.array(mask_im)

    new_im_array = np.ones_like(im_array) * bkg_fill

    # Copy this region from the blurred image on to the original
    new_im_array[mask.astype(bool)] = im_array[mask.astype(bool)]

    # Instance is most likely surrounded by whitespace. Crop such that this is removed
    if return_cropped:
        min_i = np.where(np.sum(mask, axis=1) > 0)[0][0]  # First non-zero element when summed column-wise
        min_j = np.where(np.sum(mask, axis=0) > 0)[0][0]  # First non-zero element when summed row-wise
        max_i = np.where(np.sum(mask, axis=1) > 0)[0][-1]  # Last non-zero element when summed column-wise
        max_j = np.where(np.sum(mask, axis=0) > 0)[0][-1]  # Last non-zero element when summed row-wise

        new_im_array = new_im_array[min_i:max_i, min_j:max_j]

    if return_grayscale:
        new_im_array = np.dot(new_im_array[..., :3], [0.299, 0.587, 0.114])

    try:
        new_im = Image.fromarray(new_im_array)
    except ValueError:
        print 'im_array.shape = ', im_array.shape
        print 'poly = ', poly
        print 'min_i, max_i, min_j, max_j = ', min_i, max_i, min_j, max_j
        print 'new_im_array.shape = ', new_im_array.shape
        raise

    if new_im.mode != 'RGB':
        new_im = new_im.convert('RGB')

    return new_im


def rgba_to_rgb(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318
    Source: http://www.javacms.tech/questions/56660/convert-rgba-png-to-rgb-with-pil

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background
