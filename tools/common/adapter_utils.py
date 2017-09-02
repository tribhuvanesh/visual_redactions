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

from privacy_filters.tools.common.utils import load_attributes

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def prev_to_new_attr_vec(attr_vec_v1, attr_id_to_idx_v1=None, attr_id_to_idx_v2=None):
    if attr_id_to_idx_v1 is None:
        _, attr_id_to_idx_v1 = load_attributes(v1_attributes=True)
    idx_to_attr_id_v1 = {v: k for k, v in attr_id_to_idx_v1.iteritems()}

    if attr_id_to_idx_v2 is None:
        _, attr_id_to_idx_v2 = load_attributes()
    idx_to_attr_id_v2 = {v: k for k, v in attr_id_to_idx_v2.iteritems()}

    n_attr_v1 = len(idx_to_attr_id_v1)
    n_attr_v2 = len(idx_to_attr_id_v2)

    attr_vec_v2 = np.zeros(n_attr_v2)
    attr_vec_v2[:n_attr_v1] = attr_vec_v1  # New attributes have been *appended* to the previous list

    # a105_face_all = a9_face_complete + a10_face_partial
    attr_vec_v2[attr_id_to_idx_v2['a105_face_all']] = max(attr_vec_v1[attr_id_to_idx_v1['a9_face_complete']],
                                                          attr_vec_v1[attr_id_to_idx_v1['a10_face_partial']])

    # a106_address_current_all = a74_address_current_complete + a75_address_current_partial
    attr_vec_v2[attr_id_to_idx_v2['a106_address_current_all']] = max(
        attr_vec_v1[attr_id_to_idx_v1['a74_address_current_complete']],
        attr_vec_v1[attr_id_to_idx_v1['a75_address_current_partial']])

    # a107_address_home_all = a78_address_home_complete + a79_address_home_partial
    attr_vec_v2[attr_id_to_idx_v2['a107_address_home_all']] = max(
        attr_vec_v1[attr_id_to_idx_v1['a78_address_home_complete']],
        attr_vec_v1[attr_id_to_idx_v1['a79_address_home_partial']])

    # a108_license_plate_all = a103_license_plate_complete + a104_license_plate_partial
    attr_vec_v2[attr_id_to_idx_v2['a108_license_plate_all']] = max(
        attr_vec_v1[attr_id_to_idx_v1['a103_license_plate_complete']],
        attr_vec_v1[attr_id_to_idx_v1['a104_license_plate_partial']])

    # a109_person_body = a1_age_approx + a2_weight_approx + a3_height_approx + a4_gender + a16_race + a17_color
    attr_vec_v2[attr_id_to_idx_v2['a109_person_body']] = max(
        attr_vec_v1[attr_id_to_idx_v1['a1_age_approx']],
        attr_vec_v1[attr_id_to_idx_v1['a2_weight_approx']],
        attr_vec_v1[attr_id_to_idx_v1['a3_height_approx']],
        attr_vec_v1[attr_id_to_idx_v1['a4_gender']],
        attr_vec_v1[attr_id_to_idx_v1['a16_race']],
        attr_vec_v1[attr_id_to_idx_v1['a17_color']])

    # a110_nudity_all = a12_semi_nudity + a13_full_nudity
    attr_vec_v2[attr_id_to_idx_v2['a110_nudity_all']] = max(attr_vec_v1[attr_id_to_idx_v1['a12_semi_nudity']],
                                                            attr_vec_v1[attr_id_to_idx_v1['a13_full_nudity']])

    return attr_vec_v2


def prev_to_new_masks(masks_v1, attr_id_to_idx_v1=None, attr_id_to_idx_v2=None):
    """
    Infer and append new masks
    :param masks: a 68 x X x Y matrix
    :param attr_id_to_idx_v1:
    :param attr_id_to_idx_v2:
    :return:
    """
    if attr_id_to_idx_v1 is None:
        _, attr_id_to_idx_v1 = load_attributes(v1_attributes=True)
    idx_to_attr_id_v1 = {v: k for k, v in attr_id_to_idx_v1.iteritems()}

    if attr_id_to_idx_v2 is None:
        _, attr_id_to_idx_v2 = load_attributes()
    idx_to_attr_id_v2 = {v: k for k, v in attr_id_to_idx_v2.iteritems()}

    n_attr_v1 = len(idx_to_attr_id_v1)
    n_attr_v2 = len(idx_to_attr_id_v2)

    n_new_attr = n_attr_v2 - n_attr_v1
    masks_v2 = np.concatenate((masks_v1, np.zeros((n_new_attr, masks_v1.shape[1], masks_v1.shape[2]))))

    # a105_face_all = a9_face_complete + a10_face_partial
    masks_v2[attr_id_to_idx_v2['a105_face_all']] = np.maximum(masks_v1[attr_id_to_idx_v1['a9_face_complete']],
                                                              masks_v1[attr_id_to_idx_v1['a10_face_partial']])

    # a106_address_current_all = a74_address_current_complete + a75_address_current_partial
    masks_v2[attr_id_to_idx_v2['a106_address_current_all']] = np.maximum(
        masks_v1[attr_id_to_idx_v1['a74_address_current_complete']],
        masks_v1[attr_id_to_idx_v1['a75_address_current_partial']])

    # a107_address_home_all = a78_address_home_complete + a79_address_home_partial
    masks_v2[attr_id_to_idx_v2['a107_address_home_all']] = np.maximum(
        masks_v1[attr_id_to_idx_v1['a78_address_home_complete']],
        masks_v1[attr_id_to_idx_v1['a79_address_home_partial']])

    # a108_license_plate_all = a103_license_plate_complete + a104_license_plate_partial
    masks_v2[attr_id_to_idx_v2['a108_license_plate_all']] = np.maximum(
        masks_v1[attr_id_to_idx_v1['a103_license_plate_complete']],
        masks_v1[attr_id_to_idx_v1['a104_license_plate_partial']])

    # a109_person_body = a1_age_approx + a2_weight_approx + a3_height_approx + a4_gender + a16_race + a17_color
    # np.maximum() allows comparison only between two arrays. So, iteratively cover required attributes
    # and write them in-place
    masks_v2[attr_id_to_idx_v2['a109_person_body']] = masks_v1[attr_id_to_idx_v1['a1_age_approx']].copy()
    for old_attr_id in ['a2_weight_approx', 'a3_height_approx', 'a4_gender', 'a16_race', 'a17_color']:
        np.maximum(masks_v2[attr_id_to_idx_v2['a109_person_body']], masks_v1[attr_id_to_idx_v1[old_attr_id]],
                   out=masks_v2[attr_id_to_idx_v2['a109_person_body']])

    # a110_nudity_all = a12_semi_nudity + a13_full_nudity
    masks_v2[attr_id_to_idx_v2['a110_nudity_all']] = np.maximum(masks_v1[attr_id_to_idx_v1['a12_semi_nudity']],
                                                                masks_v1[attr_id_to_idx_v1['a13_full_nudity']])

    return masks_v2
