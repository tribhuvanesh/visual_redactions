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
import copy

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread

from pycocotools import mask as mask_utils

from privacy_filters.config import *
from privacy_filters.tools.common.utils import *

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def bb_to_verts(bb):
    '''
    Convert Google BBox to convert to [ [x0, y0], [x1, y1], [x2, y2], .., [x0, y0]]
    :param bb:
    :return:
    '''
    vrts_dct = bb['vertices']  # List of [ {x:__, y:__}, ... ]
    vrts = [[d.get('x', 0), d.get('y', 0)] for d in vrts_dct]  # Convert to [ [x1, y1], [x2, y2], ..]
    vrts.append(vrts[0])  # Reconnect to first vertex
    return np.ndarray.flatten(np.asarray(vrts))


def unflatten_verts(vrts):
    '''
    [ x0, y0,  x1, y1,  x2, y2, ..,  x0, y0 ] -> [ [x0, y0], [x1, y1], [x2, y2], .., [x0, y0]]
    '''
    n_verts = len(vrts)
    _v = vrts.copy()
    _v.resize((n_verts/2, 2))
    return _v.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input annotation file")
    parser.add_argument("outfile", type=str, help="Output path")
    args = parser.parse_args()
    params = vars(args)

    infile = params['infile']
    outfile = params['outfile']

    # --- Load some necessary helpers ----------------------------------------------------------------------------------
    image_index = get_image_id_info_index()

    with open(infile) as f:
        full_anno = json.load(f)
    print '# Images found = ', len(full_anno['annotations'])

    # --- Load extra anno ----------------------------------------------------------------------------------------------
    # Create a mapping: image_id -> text_anno
    print 'Loading imageid -> text detections...'

    extra_anno_path_base = osp.join(Paths.PROJECT_ROOT, 'annotations-extra')
    image_id_to_eanno = dict()
    image_id_to_eanno_path = dict()
    for fold_name in ['train2017', 'val2017', 'test2017']:
        print 'Indexing fold: ', fold_name
        extra_anno_path = osp.join(extra_anno_path_base, fold_name)
        anno_files = os.listdir(extra_anno_path)
        for anno_file in anno_files:
            image_id, _ = osp.splitext(anno_file)
            anno_path = osp.join(extra_anno_path, anno_file)
            image_id_to_eanno_path[image_id] = anno_path
            # eanno = json.load(open(anno_path))
            # if 'fullTextAnnotation' in eanno:
            #     image_id_to_eanno[image_id] = eanno['fullTextAnnotation']

    # --- Create Sequence Labels ---------------------------------------------------------------------------------------
    anno = copy.deepcopy(full_anno['annotations'])
    for image_idx, (image_id, entry) in enumerate(tqdm(anno.items())):
        sequences = {
            'word_text': [],
            'word_attr_ids': [],  # Labels
            'word_rle': [],
            'word_bbox': [],
            'word_poly': [],
            'word_overlap_score': [],  # What fraction of this word 'W' is within gt region 'G' i.e., |W & G| / |W|
            'word_is_break': [],
        }

        entry = anno[image_id]
        eanno_path = image_id_to_eanno_path[image_id]
        eanno = json.load(open(eanno_path))
        if 'fullTextAnnotation' in eanno:
            tanno = eanno['fullTextAnnotation']
        else:
            continue

        image_width, image_height = entry['image_width'], entry['image_height']

        # Used to compute overlap scores
        gt_mat = np.zeros((image_height, image_width), dtype=np.uint8)
        wd_mat = np.zeros((image_height, image_width), dtype=np.uint8)

        for page in tanno['pages']:
            for block_idx, block in enumerate(page['blocks']):
                for par_idx, paragraph in enumerate(block['paragraphs']):
                    for word_idx, word in enumerate(paragraph['words']):
                        word_poly = bb_to_verts(word['boundingBox'])  # [ [x0, y0], [x1, y1], [x2, y2], .., [x0, y0]]
                        word_rle = mask_utils.merge(mask_utils.frPyObjects([word_poly, ], image_height, image_width))
                        word_bbox = map(int, mask_utils.toBbox(word_rle).tolist())  # (x, y, w, h)
                        word_str = ''

                        for symbol in word['symbols']:
                            symbol_str = symbol['text']
                            word_str += symbol_str

                            if 'detectedBreak' in symbol['property']:
                                # Types of line breaks: https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate#DetectedBreak
                                break_type = symbol['property']['detectedBreak']['type']
                            else:
                                break_type = None

                        # ----- Add word to sequence -----
                        # What's the label for this word? Also, what's the assignment score?
                        attr_id = SAFE_ATTR
                        overlap = 0.0

                        # Set bimask indicating this word's position
                        wd_mat[:, :] = 0  # Initialize
                        (w_x, w_y, w_w, w_h) = word_bbox
                        wd_mat[w_y:w_y + w_h, w_x:w_x + w_w] = 1
                        w_area = float(np.sum(wd_mat))  # |W|

                        # Iterate through each GT region to find best match
                        for gt in filter(lambda x: x['attr_id'] in TEXT_ATTR, entry['attributes']):
                            gt_rle = gt['segmentation']
                            gt_bbox = gt['bbox']  # (x, y, w, h)
                            gt_attr_id = gt['attr_id']
                            gt_eanno_iou = mask_utils.iou([gt_rle, ], [word_rle, ], [False, ])[0][0]

                            if gt_eanno_iou > 0.0:
                                # Bimask indicating this GT's position
                                gt_mat[:, :] = 0
                                (x, y, w, h) = map(int, gt_bbox)
                                gt_mat[y:y + h, x:x + w] = 1
                                g_and_w = float(np.sum(np.logical_and(wd_mat == 1, gt_mat == 1)))  # |W & G|
                                overlap_score = g_and_w / w_area

                                if overlap_score > overlap:
                                    attr_id = gt_attr_id
                                    overlap = overlap_score

                        sequences['word_text'].append(word_str)
                        sequences['word_attr_ids'].append(attr_id)
                        sequences['word_rle'].append(word_rle)
                        sequences['word_poly'].append(unflatten_verts(word_poly))
                        sequences['word_bbox'].append(word_bbox)
                        sequences['word_overlap_score'].append(overlap)
                        sequences['word_is_break'].append(False)

                        # ----- Optionally add line_break to sequence -----
                        if break_type is not None:
                            sequences['word_text'].append('{}'.format(break_type))
                            # Fill this in later on
                            sequences['word_attr_ids'].append(None)
                            sequences['word_rle'].append(None)
                            sequences['word_poly'].append(None)
                            sequences['word_bbox'].append(None)
                            sequences['word_overlap_score'].append(None)
                            sequences['word_is_break'].append(True)

        # Iterate over existing sequence and fill-in gaps (the Nones from breaks)
        seq_len = len(sequences['word_text'])
        for i in range(1, seq_len - 1):
            if sequences['word_attr_ids'][i] is None:
                break_type = sequences['word_text'][i]
                sequences['word_text'][i] = '<{}>'.format(break_type)

                if break_type in ['SPACE', 'SURE_SPACE']:
                    # Label this as part of the same attribute
                    label_before = sequences['word_attr_ids'][i - 1]
                    label_after = sequences['word_attr_ids'][i + 1]
                    if label_before == label_after:
                        sequences['word_attr_ids'][i] = label_before
                    else:
                        sequences['word_attr_ids'][i] = SAFE_ATTR

                    # Infer RLE for this whitespace
                    poly_before = sequences['word_poly'][i - 1]  # [ [x0, y0], [x1, y1], [x2, y2], .., [x0, y0]]
                    poly_after = sequences['word_poly'][i + 1]
                    if poly_before is not None and poly_after is not None:
                        # (x, y) is top-left. So, use top-right of before bbox
                        tl = poly_before[1]
                        tr = poly_after[0]
                        br = poly_after[3]
                        bl = poly_before[2]
                        ws_poly = [tl, tr, br, bl, tl]
                        _ws_poly = np.ndarray.flatten(np.asarray(ws_poly))
                        ws_rle = mask_utils.merge(mask_utils.frPyObjects([_ws_poly, ], image_height, image_width))
                        sequences['word_rle'][i] = ws_rle
                        sequences['word_poly'][i] = ws_poly
                        sequences['word_bbox'][i] = map(int, mask_utils.toBbox(ws_rle).tolist())  # (x, y, w, h)
                        sequences['word_overlap_score'][i] = (sequences['word_overlap_score'][i - 1] +
                                                              sequences['word_overlap_score'][i + 1]) / 2.0
                else:
                    sequences['word_attr_ids'][i] = SAFE_ATTR

        # Iterate once again. But now, expand the poly to end at beginning of next word (since there are spaces between words)
        for i in range(0, seq_len - 1):
            if sequences['word_is_break'][i] and sequences['word_text'][i] not in ['<SPACE>', '<SURE_SPACE>']:
                # Do nothing if this word is a line break
                continue
            if sequences['word_poly'][i + 1] is None:
                # Also, do nothing if we don't have the adjacent polygon
                continue

            # Otherwise, extend this word region until the next word
            cur_poly = sequences['word_poly'][i]  # [ [x0, y0], [x1, y1], [x2, y2], .., [x0, y0]]
            next_poly = sequences['word_poly'][i + 1]

            # Top-right is Top-left of next word
            cur_poly[1] = next_poly[0]
            # Bottom-right is Bottom-left of next word
            cur_poly[2] = next_poly[3]

            sequences['word_poly'][i] = cur_poly

            # Re-compute RLE
            _poly = np.ndarray.flatten(np.asarray(cur_poly))
            _rle = mask_utils.merge(mask_utils.frPyObjects([_poly, ], image_height, image_width))
            sequences['word_rle'][i] = _rle

        # Handle first and last tokens separately
        if sequences['word_attr_ids'][0] is None:
            sequences['word_attr_ids'][0] = SAFE_ATTR
            sequences['word_text'][0] = '<{}>'.format(sequences['word_text'][0])
        if sequences['word_attr_ids'][-1] is None:
            sequences['word_attr_ids'][-1] = SAFE_ATTR
            sequences['word_text'][-1] = '<{}>'.format(sequences['word_text'][-1])

        entry['sequence'] = sequences
        del entry['attributes']

    # --- Write
    anno = dict(filter(lambda x: 'sequence' in x[1], anno.iteritems()))

    print 'Writing {} image sequences to {}'.format(len(anno), outfile)

    with open(outfile, 'w') as wf:
        json.dump(anno, wf, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()