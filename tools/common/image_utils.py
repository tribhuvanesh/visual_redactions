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

from skimage.segmentation import slic, mark_boundaries

from scipy.misc import imread, imresize

from pycocotools import mask as mask_utils

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
        new_h = int(np.round(h * (new_w / float(w))))  # Scale height to same aspect ratio
    else:
        new_h = mins_len
        new_w = int(np.round(w * (new_h / float(h))))  # Scale height to same aspect ratio
    return pil_img.resize((new_w, new_h))


def bimask_to_rgba(bimask, color=np.array([95, 242, 186])):
    h, w = bimask.shape
    img_arr = np.zeros((h, w, 4))

    # Set alpha
    img_arr[:, :, -1] = bimask
    img_arr[:, :, :3] = color

    # return Image.fromarray(img_arr.astype('uint8'))
    return img_arr


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


def get_instance_crop(img_path, rle, bbox=None):
    """

    :param img_path: Absolute path to image
    :param rle: RLE-encoded instance
    :param bbox: [x, y, w, h]
    :return:
    """
    if bbox is None:
        bbox = mask_utils.toBbox(rle)

    im = Image.open(img_path).convert('RGB')
    imarr = np.asarray(im).copy()
    bimask = mask_utils.decode(rle)
    bimask = np.tile(bimask[:, :, None], 3)  # RGB
    imarr[bimask == 0] = 255  # Set pixels outside instance as white

    x, y, w, h = bbox
    masked_im = Image.fromarray(imarr).crop([x, y, x + w, y + h])

    del im
    del imarr
    del bimask

    return masked_im


def redact_img(pil_img, segmentation, fill='black', outline='black'):
    if type(segmentation) is not list:
        raise NotImplementedError
    else:
        polys = segmentation
        if type(polys[0]) is not list:
            polys = [polys, ]
        im = pil_img.copy()
        draw = ImageDraw.Draw(im)
        for poly in polys:
            draw.polygon(poly, fill=fill, outline=outline)
        del draw
        return im


def redact_img_mask(pil_img, bimask):
    imarr = np.asarray(pil_img).copy()
    mask3d = np.tile(bimask[:, :, None], 3)
    imarr[np.where(bimask == 1)] = 0
    im = Image.fromarray(imarr)
    del mask3d
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


def seg_to_adj(X):
    """
    Convert a matrix of labels to an adjacency matrix
    https://stackoverflow.com/questions/26486898/matrix-of-labels-to-adjacency-matrix
    :param X: Matrix of labels (like ones produced by SLIC)
    :return: Adjacency matrix
    """
    n = len(np.unique(X))
    G = np.zeros((n, n), dtype=np.int)

    # left-right pairs
    G[X[:, :-1], X[:, 1:]] = 1
    # right-left pairs
    G[X[:, 1:], X[:, :-1]] = 1
    # top-bottom pairs
    G[X[:-1, :], X[1:, :]] = 1
    # bottom-top pairs
    G[X[1:, :], X[:-1, :]] = 1

    return G


def dilate_mask(_seg, _mask, c):
    """
    Dilates bimask by a factor of c given SLIC label assignment mantrix _seg
    :param _seg: N x M superpixel assignment matrix
    :param _mask:  N x M binary mask
    :param c: dilation factor (>= 1.0)
    :return: N x M dilated binary mask
    """
    if c < 1.0:
        raise ValueError('c needs to be >=1.0')

    _mask = _mask.copy()

    cur_pixels = np.sum(_mask)
    target_pixels = min(c * cur_pixels, _mask.size)

    vrts_all = set(np.unique(_seg))
    vrts_in_mask = set(np.unique(_mask * _seg))  # in-vert

    # First add all segments containing seed pixels with >25% overlap
    # overlap_vrts = np.unique(segments * new_bimask)
    for _v in list(vrts_in_mask):
        # Add this vertex only if it overlaps > 25%
        # Pixels in this superpixel
        n_sup_pix = np.sum(_seg == _v)
        # Pixels in overlap
        n_overlap = np.sum(np.logical_and(_mask == 1, _seg == _v)).astype(np.float32)

        if n_overlap / n_sup_pix > 0.25:
            _mask[np.where(_seg == _v)] = 1.0
        else:
            vrts_in_mask.remove(_v)

    cur_pixels = np.sum(_mask)
    A = seg_to_adj(_seg)
    while cur_pixels < target_pixels:
        # for _i in range(2):
        vrts_outside_mask = vrts_all - vrts_in_mask  # out-vert

        # Choose a single vrt from vrts_outside_mask to add to set
        # For each out-vert get a count of how many edges it has to an in-vert
        candidates = []  # List of (_v, _ne) where _ne = # edges with an in-vert
        for _v in vrts_outside_mask:
            adj_vrts = set(np.where(A[_v] > 0)[0])
            _ne = len(vrts_in_mask & adj_vrts)
            candidates.append((_v, _ne))

        # Choose the best candidate
        candidates = sorted(candidates, key=lambda x: -x[1])
        max_ne = np.max(map(lambda x: x[1], candidates))  # What's the highest no. of edges for any node?
        candidates = filter(lambda x: x[1] == max_ne, candidates)  # Filter vertices with these many edges
        candidates_v = [x[0] for x in candidates]

        best_v = np.random.choice(candidates_v)
        vrts_in_mask.add(best_v)

        # Add this vertex to mask
        _mask[np.where(_seg == best_v)] = 1.0
        cur_pixels = np.sum(_mask)

    return _mask


def contract_mask(_seg, _mask, c):
    """
    Contracts bimask by a factor of c given SLIC label assignment mantrix _seg.
    This is simply an inverse dilation problem. So, we perform dilation on an inverted mask.
    :param _seg: N x M superpixel assignment matrix
    :param _mask:  N x M binary mask
    :param c: dilation factor (>= 1.0)
    :return: N x M dilated binary mask
    """
    if c > 1.0:
        raise ValueError('c needs to be <=1.0')

    cur_pixels = np.sum(_mask)
    target_pixels = c * cur_pixels

    img_area = float(_mask.size)

    # What is c in terms of #0s in the mask?
    inv_mask = (_mask == 0).astype(int)
    new_c = (img_area - target_pixels) / float(np.sum(inv_mask))
    # print c, new_c, np.sum(_mask), np.sum(inv_mask)

    new_inv_mask = dilate_mask(_seg, inv_mask, new_c)
    new_mask = (new_inv_mask == 0).astype(np.uint8)

    # print c, new_c, np.sum(_mask) / img_area, np.sum(new_mask) / img_area, np.unique(new_mask)

    return new_mask


def resize_bimask(bimask, max_len=1000.):
    org_h, org_w = bimask.shape
    max_len = float(max_len)

    if org_w > org_h:
        new_w = max_len
        new_h = (new_w / org_w) * org_h
    else:
        new_h = max_len
        new_w = (new_h / org_h) * org_w
    new_w, new_h = int(new_w), int(new_h)

    new_bimask = imresize(bimask, (new_h, new_w)).astype(bimask.dtype)
    return new_bimask


def scale_mask(im, bimask, c, n_segments=200, smoothen=True):
    """
    Scale bimask for image im by a factor c
    :param im:
    :param bimask:
    :param c:
    :return:
    """
    # Resize image so that SLIC is faster
    # Resize image to a lower size
    org_w, org_h = im.size
    max_len = 1000.

    if org_w > org_h:
        new_w = max_len
        new_h = (new_w / org_w) * org_h
    else:
        new_h = max_len
        new_w = (new_h / org_h) * org_w
    new_w, new_h = int(new_w), int(new_h)

    new_im = im.resize((new_w, new_h))
    new_bimask = imresize(bimask, (new_h, new_w))

    segments = slic(new_im, n_segments=n_segments, slic_zero=True)

    if c > 1.:
        scaled_mask = dilate_mask(segments, new_bimask, c)
    elif c < 1.:
        scaled_mask = contract_mask(segments, new_bimask, c)
    else:
        scaled_mask = new_bimask

    if smoothen:
        smooth_bimask = Image.fromarray(scaled_mask.astype('uint8') * 255)
        for i in range(10):
            smooth_bimask = smooth_bimask.filter(ImageFilter.GaussianBlur)
        scaled_mask = np.asarray(smooth_bimask) > 128
        scaled_mask = scaled_mask.astype('uint8')

    # Rescale this mask to original size
    scaled_mask_highres = imresize(scaled_mask, (org_h, org_w), interp='nearest')

    del new_im
    del new_bimask
    del scaled_mask

    return scaled_mask_highres
