#!/usr/bin/python
"""Given a GT and a Prediction file, evaluate predictions

Here, we want simple predictions, per:
- image     (N images)
- attribute (K attributes)

the metrics:
- precision
- recall
- IoU

So, construct three NxK matrices for each of these
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
from collections import defaultdict as dd
import datetime

import numpy as np
import matplotlib.pyplot as plt

from pycocotools import mask as mask_utils

from PIL import Image
from scipy.misc import imread, imresize

import pprint

from privacy_filters.tools.common.image_utils import bimask_to_rgba
from privacy_filters.tools.common.utils import load_attributes_shorthand, get_image_id_info_index
from privacy_filters.tools.evaltools.evaltools import compute_eval_metrics
from privacy_filters.config import *

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class VISPRSegEvalSimple:
    """
    GT format:
    {
        #--------- One per anno file ---------
        'created_at: '2017-08-29 15:25:11.001926',
        'stats':   { ..... },
        'annotations': {
         #--------- One per image ---------
         '2017_235123' :   {
                'image_id': '2017_235123',
                'image_path': 'images/val2017/2017_18072751.jpg'
                'image_height': 1024,
                'image_width' : 2048,
                'attributes': [     #--------- One per instance ---------
                    {
                        'instance_id':           4,
                        'attr_id':      'a105_face_all',
                        'polygons': [[], ],          # polygon [[x1 y1 x2 y2, ...], [x1 y1 x2 y2, ...], ]
                        'area':         [float, ...],     # One per region (instance can exist as multiple regions)
                        'bbox':         [[x, y, width, height], ...]   # One per region
                        'segmentation': RLE  # polygons encoded as RLE (see MS-COCO format)
                    }
                ]
            }
        }
    }

    Prediction file format:
    (Almost the same as COCO segmentation format: http://cocodataset.org/dataset.htm#format)
    [
        {
            'image_id': '2017_235123',
            'attr_id': 'a105_face_all',
            'segmentation': RLE,
            'score': float,
        }
    """
    def __init__(self, gt_path, pred_path):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.vispr_gt_full = json.load(open(gt_path))
        self.vispr_gt = self.vispr_gt_full['annotations']
        self.vispr_pred = json.load(open(pred_path))

        self.evalImgs = dict()    # Three matrices of size NxK: precision, recall, iou
        self.eval = {}  # accumulated evaluation results

        self._gts = dd(list)   # Map (image_id, attr_id) -> [gt_detections, ]
        self._pds = dd(list)   # Map (image_id, attr_id) -> [detections, ]
        self.ious = {}         # Map (image_id, attr_id) -> IoU matrix (preds x gt)

        self.params = Params()
        self.params.imgIds = sorted(np.unique(self.vispr_gt.keys()))
        self.params.attrIds = sorted(np.unique(self.vispr_gt_full['stats']['present_attr']))
        self._paramsEval = {}  # parameters for evaluation

        # Remove attributes which need to be ignored
        self.params.attrIds = [x for x in self.params.attrIds if x not in IGNORE_ATTR]

        pred_imgIds = np.unique([e['image_id'] for e in self.vispr_pred])
        print '# Predicted Images = ', len(pred_imgIds)
        print '# GT Images = ', len(self.params.imgIds)
        print '# Common = ', len(set(pred_imgIds) & set(self.params.imgIds))
        print '# Attributes = ', len(self.params.attrIds)

        # Load Attributes to shorthand
        self.attr_id_to_name = load_attributes_shorthand()
        self.attr_names = [self.attr_id_to_name[attr_id] for attr_id in self.params.attrIds]

        self.image_id_index = get_image_id_info_index()

        self.overall_stats = dict()
        self.stats_str = ""

        # Setup colors
        np.random.seed(42)
        self.colors = [(np.random.random(size=3) * 255).astype(int) for i in range(40)]

    def prepare(self):
        """
        Populate _gts and _pds
        :return:
        """
        # --- Prepared GT ----------------------------------------------------------------------------------------------
        next_gt_id = 0
        for image_id, anno_entry in self.vispr_gt.iteritems():
            image_height, image_width = anno_entry['image_height'], anno_entry['image_width']
            for gt in anno_entry['attributes']:
                if gt.get('segmentation', None) is None:
                    # Obtain RLE of mask if this doesn't already exist
                    rles = mask_utils.frPyObjects(gt['polygons'], image_height, image_width)
                    rle = mask_utils.merge(rles)
                    gt['segmentation'] = rle
                del gt['polygons']  # Free memory
                gt['id'] = '{}_{}'.format(image_id, gt['instance_id'])
                gt['id'] = int(gt['id'].replace('_', ''))
                # gt['id'] = next_gt_id
                # next_gt_id += 1
                gt['iscrowd'] = gt.get('iscrowd', 0)
                gt['area'] = np.sum(gt['area'])
                gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
                gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
                attr_id = gt['attr_id']
                self._gts[(image_id, attr_id)].append(gt)

        # --- Prepared Predictions -------------------------------------------------------------------------------------
        next_pred_id = 0
        next_pred_id_dd = dd(int)
        n_pred_ignored = 0
        for pred in self.vispr_pred:
            if pred['score'] < self.params.score_thresh:
                n_pred_ignored += 1
                continue
            image_id = pred['image_id']
            attr_id = pred['attr_id']
            assert pred.get('segmentation', None) is not None
            # pred['id'] = next_pred_id
            # next_pred_id += 1
            pred['id'] = '{}_{}'.format(image_id, next_pred_id_dd[image_id])
            pred['id'] = int(pred['id'].replace('_', ''))
            next_pred_id_dd[image_id] += 1
            pred['area'] = mask_utils.area(pred['segmentation'])
            self._pds[(image_id, attr_id)].append(pred)

        print '# Predictions available = ', len(self.vispr_pred)
        print '# Ignored (score < {}) = {}'.format(self.params.score_thresh, n_pred_ignored)

        self.evalImgs = dd(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self, visualize_dir=None):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params

        p.imgIds = list(np.unique(p.imgIds))
        self.params = p

        self.prepare()
        attr_ids = p.attrIds

        n_imgs = len(p.imgIds)
        n_attr = len(attr_ids)

        tp = -np.ones((n_imgs, n_attr), dtype=np.float32)
        fp = -np.ones((n_imgs, n_attr), dtype=np.float32)
        fn = -np.ones((n_imgs, n_attr), dtype=np.float32)

        rel_sizes = -np.ones((n_imgs, n_attr), dtype=np.float32)

        precision = -np.ones((n_imgs, n_attr), dtype=np.float32)
        recall    = -np.ones((n_imgs, n_attr), dtype=np.float32)
        iou       = -np.ones((n_imgs, n_attr), dtype=np.float32)

        n_masks_ignored = 0
        n_masks_used = 0

        if visualize_dir:
            # Create two directories -
            # a. one for summary (all gt/prediction masks)
            # b. one for per mask (i.e., for each gt, prediction pair, create an image)
            visualize_summary_dir = osp.join(visualize_dir, 'summary')
            visualize_attr_dir = osp.join(visualize_dir, 'per_attr')

            for _d in [visualize_summary_dir, visualize_attr_dir]:
                if not osp.exists(_d):
                    print 'Path {} does not exist. Creating it...'.format(_d)
                    os.makedirs(_d)

        for image_idx, image_id in enumerate(p.imgIds):
            sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (image_idx, n_imgs,
                                                                    (image_idx * 100.0) / n_imgs))
            sys.stdout.flush()

            img_w, img_h = self.vispr_gt[image_id]['image_width'], self.vispr_gt[image_id]['image_height']
            img_area = float(img_w * img_h)  # No. of pixels in this image

            if visualize_dir is not None:
                mask_stats_list = []  # (attr_id, gt_mask, pred_mask, prec, rec, iou)
            else:
                mask_stats_list = None

            vis_img = False

            for attr_idx, attr_id in enumerate(attr_ids):
                key = (image_id, attr_id)
                gt_exists = key in self._gts
                pd_exists = key in self._pds

                if (not gt_exists) and (not pd_exists):
                    continue

                # GT RLE
                gt = self._gts[(image_id, attr_id)]
                gt_rle = self.dets_to_rle(gt, (img_w, img_h))
                # gt_rle = None if gt_rle is None else [gt_rle, ]

                if gt_rle is not None and mask_utils.area(gt_rle) < MIN_PIXELS:
                    n_masks_ignored += 1
                    continue
                elif gt_rle is not None:
                    n_masks_used += 1

                # Predicted RLE
                pd = self._pds[(image_id, attr_id)]
                pd_rle = self.dets_to_rle(pd, (img_w, img_h))
                # pd_rle = None if pd_rle is None else [pd_rle, ]

                if gt_rle is None:
                    n_ones = mask_utils.area(pd_rle)
                    n_zeros = img_area - n_ones
                    _tp, _fp, _tn, _fn = 0, n_ones, n_zeros, 0
                elif pd_rle is None:
                    n_ones = mask_utils.area(gt_rle)
                    n_zeros = img_area - n_ones
                    _tp, _fp, _tn, _fn = 0, 0, n_zeros, n_ones
                else:
                    # FIXME Ignoring crowd masks for now
                    # _tp, _fp, _tn, _fn = mask_utils.get_metrics(gt_rle, pd_rle)
                    gt_mask = mask_utils.decode(gt_rle)
                    pd_mask = mask_utils.decode(pd_rle)
                    _, _, _, _tp, _fp, _fn = compute_eval_metrics(gt_mask, pd_mask)
                    del gt_mask
                    del pd_mask

                # Precision
                if (_tp + _fp) > 0.0:
                    _prec = float(_tp) / (_tp + _fp)
                else:
                    _prec = np.nan

                # Recall
                if (_tp + _fn) > 0.0:
                    _rec = float(_tp) / (_tp + _fn)
                else:
                    _rec = np.nan

                # IoU
                if (_tp + _fp + _fn) > 0.0:
                    _iou = float(_tp) / (_tp + _fp + _fn)
                else:
                    _iou = np.nan

                precision[image_idx, attr_idx] = _prec
                recall[image_idx, attr_idx] = _rec
                iou[image_idx, attr_idx] = _iou

                if gt_rle is not None:
                    rel_sizes[image_idx, attr_idx] = mask_utils.area(gt_rle) / img_area

                # Normalized by image area
                tp[image_idx, attr_idx] = _tp / img_area
                fp[image_idx, attr_idx] = _fp / img_area
                fn[image_idx, attr_idx] = _fn / img_area

                if mask_stats_list is not None:
                    empty = np.zeros((img_h, img_w))
                    gt_mask = mask_utils.decode(gt_rle) if gt_rle is not None else empty
                    pd_mask = mask_utils.decode(pd_rle) if pd_rle is not None else empty
                    mask_stats_list.append((
                        attr_id,
                        gt_mask, pd_mask,
                        _prec, _rec, _iou,
                    ))

                if attr_id in MODE_TO_ATTR_ID['multimodal']:
                    vis_img = True

            if mask_stats_list is not None and vis_img:
                # Visualize summary
                vis_out_path = osp.join(visualize_summary_dir, image_id + '.jpg')
                self.visualize_img_summary(mask_stats_list, image_id, vis_out_path)

                # Visualize per attribute mask
                self.visualize_attr_stats(mask_stats_list, image_id, visualize_attr_dir)
                del mask_stats_list

        self.evalImgs['precision'] = precision
        self.evalImgs['recall'] = recall
        self.evalImgs['iou'] = iou

        self.evalImgs['rel_sizes'] = rel_sizes

        self.evalImgs['tp'] = tp
        self.evalImgs['fp'] = fp
        self.evalImgs['fn'] = fn

        toc = time.time()
        print('DONE (t={:0.2f}s)  # Masks ignored = {}/{}'.format(toc - tic, n_masks_ignored, (n_masks_used+n_masks_ignored)))

    def visualize_img_summary(self, mask_stats_list, image_id, vis_out_path):
        """
        Visualizes summary for entire image -- for each gt/pred mask, visualize masks and write (P, R, IoU) scores
        :param mask_stats_list:  List of (attr_id, gt_mask, pred_mask, prec, rec, iou)
        :param vis_out_path:
        :return:
        """
        mask_stats_list = sorted(mask_stats_list, key=lambda x: x[0])  # Sort by attr_id
        n_attr = len(mask_stats_list)
        nrows = n_attr + 1
        ncols = 3   # [  GT  |  Pred  |  Stats  ]

        fig_width = ncols * 7
        fig_height = nrows * 4

        fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(fig_width, fig_height))

        im = Image.open(self.image_id_index[image_id]['image_path'])
        w, h = im.size

        # Disable axis everywhere ---------------------------
        for i in range(axarr.shape[0]):
            for j in range(axarr.shape[1]):
                axarr[i, j].axis('off')

        # First row = image ---------------------------------
        ax = axarr[0, 0]
        ax.imshow(im)

        for _idx, (attr_id, gt_mask, pred_mask, _prec, _rec, _iou, _util) in enumerate(mask_stats_list):
            row_idx = _idx + 1

            # Plot GT  --------------------------------------
            col_idx = 0
            ax = axarr[row_idx, col_idx]
            ax.imshow(im, alpha=0.5)
            if _idx == 0:
                ax.set_title('---- GT ----\n{}'.format(self.attr_id_to_name[attr_id]))
            else:
                ax.set_title('{}'.format(self.attr_id_to_name[attr_id]))
            gt_col_mask = bimask_to_rgba(gt_mask, self.colors[0])
            ax.imshow(gt_col_mask, alpha=0.8)
            del gt_mask
            del gt_col_mask

            # Plot Pred --------------------------------------
            col_idx = 1
            ax = axarr[row_idx, col_idx]
            ax.imshow(im, alpha=0.5)
            if _idx == 0:
                ax.set_title('---- Pred ----\n{}'.format(self.attr_id_to_name[attr_id]))
            else:
                ax.set_title('{}'.format(self.attr_id_to_name[attr_id]))
            pred_col_mask = bimask_to_rgba(pred_mask, self.colors[0])
            ax.imshow(pred_col_mask, alpha=0.8)
            del pred_mask
            del pred_col_mask

            # Write Stats --------------------------------------
            col_idx = 2
            ax = axarr[row_idx, col_idx]
            # ax.set_xlim([0, 1])
            # ax.set_ylim([0, 1])
            text_str = "precision: {:.2f}\nrecall: {:.2f}\niou: {:.2f}".format(_prec*100, _rec*100,
                                                                                             _iou*100)
            ax.text(0.1, 0.5, text_str, fontsize='large')

        plt.tight_layout()
        plt.savefig(vis_out_path, bbox_inches='tight')
        plt.close()

    def visualize_attr_stats(self, mask_stats_list, image_id, vis_out_path):
        """
        Visualize masks per image. Additionally, sort by predictions by score.
        :param mask_stats_list:  List of (attr_id, gt_mask, pred_mask, prec, rec, iou)
        :param vis_out_path:
        :return:
        """
        mask_stats_list = sorted(mask_stats_list, key=lambda x: x[0])  # Sort by attr_id
        im = Image.open(self.image_id_index[image_id]['image_path'])
        w, h = im.size

        for _idx, (attr_id, gt_mask, pred_mask, _prec, _rec, _iou, _util) in enumerate(mask_stats_list):

            if attr_id not in MODE_TO_ATTR_ID['multimodal']:
                continue

            fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))

            # --- GT Mask ----------------------------------------------------
            ax = axarr[0]
            ax.imshow(im, alpha=0.5)
            gt_col_mask = bimask_to_rgba(gt_mask, self.colors[0])
            ax.imshow(gt_col_mask, alpha=0.8)
            ax.axis('off')

            # --- Pred Mask --------------------------------------------------
            ax = axarr[1]
            ax.imshow(im, alpha=0.5)
            pred_col_mask = bimask_to_rgba(pred_mask, self.colors[0])
            ax.imshow(pred_col_mask, alpha=0.8)
            ax.axis('off')

            ax.set_title('{}:   P = {:.2f}    R = {:.2f}    IoU = {:.2f}'.format(self.attr_id_to_name[attr_id],
                                                                                        _prec, _rec, _iou))

            # --- Save Figure ------------------------------------------------
            vis_attr_out_path = osp.join(vis_out_path, attr_id)
            for _d in ['FN', 'FP', 'FI']:
                _od = osp.join(vis_attr_out_path, _d)
                if not osp.exists(_od):
                    os.makedirs(_od)
            if _prec is np.nan:
                # Precision is undefined (i.e., GT contains mask, Pred does not)
                im_name = '{}.jpg'.format(image_id)
                vis_im_out_path = osp.join(vis_attr_out_path, 'FN', im_name)
            elif _rec is np.nan:
                # Recall is undefined (i.e., GT does not contain mask, Pred does)
                im_name = '{}.jpg'.format(image_id)
                vis_im_out_path = osp.join(vis_attr_out_path, 'FP', im_name)
            elif _iou is np.nan:
                # IoU is undefined (this should not occur, but handle it anyway)
                im_name = '{}.jpg'.format(image_id)
                vis_im_out_path = osp.join(vis_attr_out_path, 'FI', im_name)
            else:
                # Valid Precision and Recall
                im_name = '{:.0f}_{}.jpg'.format((_iou * 10**4), image_id)
                vis_im_out_path = osp.join(vis_attr_out_path, im_name)

            plt.tight_layout()
            plt.savefig(vis_im_out_path, bbox_inches='tight')
            plt.close()

    def anno_to_bimask(self, dets, img_size, min_side_length=400.0):
        """
        Given a list of detections, combines them into a single (normalized) binary mask
        Returns a zero-filled matrix if len(dets) == 0
        :param dets:
        :param normalize:
        :return:
        """
        org_w, org_h = img_size
        if org_h > org_w:
            new_w = min_side_length
            new_h = org_h * (min_side_length / org_w)
        else:  # org_w >= org_h
            new_h = min_side_length
            new_w = org_w * (min_side_length / org_h)
        new_w, new_h = int(new_w), int(new_h)

        rle_list = []
        crowd_rle_list = []
        for det in dets:
            rle = det['segmentation']
            assert rle['size'] == [org_h, org_w]
            if det.get('ignore', False):
                crowd_rle_list.append(rle)
            else:
                rle_list.append(rle)

        if len(rle_list) > 0:
            # try:
            highres_bimask = mask_utils.decode(rle_list)[:, :, 0]
            norm_bimask = imresize(highres_bimask, (new_h, new_w), interp='nearest')
            del highres_bimask
        else:
            norm_bimask = np.zeros((new_h, new_w))

        if len(crowd_rle_list) > 0:
            highres_ig_bimask = mask_utils.decode(crowd_rle_list)
            norm_ig_bimask = imresize(highres_ig_bimask, (new_h, new_w), interp='nearest')
            del highres_ig_bimask
        else:
            norm_ig_bimask = None

        return norm_bimask, norm_ig_bimask

    def dets_to_rle(self, dets, img_size):
        org_w, org_h = img_size

        rle_list = []
        crowd_rle_list = []
        for det in dets:
            rle = det['segmentation']
            if rle['size'] != [org_h, org_w]:
                # FIXME Hack
                rle['size'] = rle['size'][::-1]
            assert rle['size'] == [org_h, org_w], "{} != {}".format(rle['size'], [org_h, org_w])
            if det.get('ignore', False):
                crowd_rle_list.append(rle)
            else:
                rle_list.append(rle)

        if len(rle_list) > 0:
            return mask_utils.merge(rle_list)
        else:
            return None

    def anno_to_bimask_org(self, dets, img_size):
        """
        Given a list of detections, combines them into a single (non-normalized) binary mask
        Returns a zero-filled matrix if len(dets) == 0
        :param dets:
        :param normalize:
        :return:
        """
        org_w, org_h = img_size

        rle_list = []
        crowd_rle_list = []
        for det in dets:
            rle = det['segmentation']
            if rle['size'] != [org_h, org_w]:
                # FIXME Hack
                rle['size'] = rle['size'][::-1]
            assert rle['size'] == [org_h, org_w], "{} != {}".format(rle['size'], [org_h, org_w])
            if det.get('ignore', False):
                crowd_rle_list.append(rle)
            else:
                rle_list.append(rle)

        if len(rle_list) > 0:
            # try:
            highres_bimask = mask_utils.decode(mask_utils.merge(rle_list))
        else:
            highres_bimask = np.zeros((org_h, org_w))

        if len(crowd_rle_list) > 0:
            highres_ig_bimask = mask_utils.decode(mask_utils.merge(crowd_rle_list))
        else:
            highres_ig_bimask = None

        return highres_bimask, highres_ig_bimask

    def accumulate(self):
        tic = time.time()
        print('Accumulating results...')
        p = self.params
        precision = self.evalImgs['precision']  # N x K matrix
        recall = self.evalImgs['recall']
        iou = self.evalImgs['iou']

        tp = self.evalImgs['tp']  # N x K matrix
        fp = self.evalImgs['fp']
        fn = self.evalImgs['fn']
        pos = tp + fp

        # Calculate per-class stats
        overall_precision = np.zeros(len(p.attrIds))
        overall_recall = np.zeros(len(p.attrIds))
        overall_iou = np.zeros(len(p.attrIds))

        overall_tp = np.zeros(len(p.attrIds))
        overall_fp = np.zeros(len(p.attrIds))
        overall_fn = np.zeros(len(p.attrIds))

        overall_pos = np.zeros(len(p.attrIds))

        for attr_idx, attr_id in enumerate(p.attrIds):
            # Precision, Recall, IoU averaged per image
            # m = precision[:, attr_idx]
            # overall_precision[attr_idx] = np.mean(m[m > -1])
            # m = recall[:, attr_idx]
            # overall_recall[attr_idx] = np.mean(m[m > -1])
            # m = iou[:, attr_idx]
            # overall_iou[attr_idx] = np.mean(m[m > -1])

            # TP, FP, FN computed per image, then averaged
            m = tp[:, attr_idx]
            overall_tp[attr_idx] = np.mean(m[m > -1])
            m = fp[:, attr_idx]
            overall_fp[attr_idx] = np.mean(m[m > -1])
            m = fn[:, attr_idx]
            overall_fn[attr_idx] = np.mean(m[m > -1])
            m = pos[:, attr_idx]
            overall_pos[attr_idx] = np.mean(m[m > -1])

            if (overall_tp[attr_idx] + overall_fp[attr_idx]) > 0:
                overall_precision[attr_idx] = overall_tp[attr_idx] / (overall_tp[attr_idx] + overall_fp[attr_idx])
            else:
                overall_precision[attr_idx] = 0.0
            overall_recall[attr_idx] = overall_tp[attr_idx] / (overall_tp[attr_idx] + overall_fn[attr_idx])
            overall_iou[attr_idx] = overall_tp[attr_idx] / (overall_tp[attr_idx] + overall_fp[attr_idx] + overall_fn[attr_idx])

        self.overall_stats['precision'] = overall_precision
        self.overall_stats['recall'] = overall_recall
        self.overall_stats['iou'] = overall_iou
        self.overall_stats['positives'] = overall_pos

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        print 'Mean Precision = ', np.mean(self.overall_stats['precision'])
        print 'Mean Recall = ', np.mean(self.overall_stats['recall'])
        print 'Mean IoU = ', np.mean(self.overall_stats['iou'])

        self.stats = np.zeros(3)
        self.stats[0] = np.mean(self.overall_stats['precision'])
        self.stats[1] = np.mean(self.overall_stats['recall'])
        self.stats[2] = np.mean(self.overall_stats['iou'])


class Params:
    """
    Adapted from coco evaluation api
    """
    def setDetParams(self):
        self.imgIds = []
        self.attrIds = []

    def __init__(self):
        self.setDetParams()
        self.score_thresh = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", type=str, help="GT File")
    parser.add_argument("pred_file", type=str, help="Predicted file")
    parser.add_argument("-r", "--row", action='store_true', default=False,
                        help="Print an additional row to aid pasting results into a spreadsheet")
    parser.add_argument("-w", "--write_results", action='store_true', default=False,
                        help="Write summarized results into JSON in predicted file directory")
    parser.add_argument("-v", "--visualize_results", type=str, default=None,
                        help="Write visualizations in this file directory")
    parser.add_argument("-t", "--threshold", type=float, default=0.0,
                        help="Use only predictions whose scores are above this threshold")
    args = parser.parse_args()
    params = vars(args)
    print
    print 'Evaluating: ', params['pred_file']
    vispr = VISPRSegEvalSimple(params['gt_file'], params['pred_file'])
    vispr.params.score_thresh = params['threshold']
    vispr.evaluate(visualize_dir=params['visualize_results'])
    vispr.accumulate()
    vispr.summarize()

    if params['row']:
        print
        # You can now copy-paste this line into a spreadsheet. Seems like this does not work from within tmux.
        print 'Overall scores: '
        print '\t'.join(map(lambda x: '{}'.format(x), vispr.stats.tolist()))
        print 'Classes: '
        print '\t'.join(vispr.attr_names)
        print 'Class Precision: '
        print '\t'.join(map(lambda x: '{}'.format(x), vispr.overall_stats['precision'].tolist()))
        print 'Class Recall: '
        print '\t'.join(map(lambda x: '{}'.format(x), vispr.overall_stats['recall'].tolist()))
        print 'Class IoU: '
        print '\t'.join(map(lambda x: '{}'.format(x), vispr.overall_stats['iou'].tolist()))

    if params['write_results']:
        pred_path, pred_nameext = osp.split(params['pred_file'])
        pred_name, _ext = osp.splitext(pred_nameext)
        out_name = pred_name + '-summary.json'
        out_path = osp.join(pred_path, out_name)

        print 'Writing results to: ', out_path

        idx_to_attr_id = dict()
        for attr_idx, attr_id in enumerate(vispr.params.attrIds):
            idx_to_attr_id[attr_idx] = attr_id
        attr_id_to_idx = {v:k for k, v in idx_to_attr_id.iteritems()}

        out_dct = dict()

        # Metadata
        out_dct['threshold'] = params['threshold']
        out_dct['gt_file'] = params['gt_file']
        out_dct['pred_file'] = params['pred_file']
        out_dct['created_at'] = str(datetime.datetime.now())

        # --- Overall stats
        out_dct['overall'] = {
            'precision': np.mean(vispr.overall_stats['precision']),
            'recall': np.mean(vispr.overall_stats['recall']),
            'iou': np.mean(vispr.overall_stats['iou']),
        }

        # --- Per Size
        out_dct['per_size'] = dict()
        for _size in SIZE_TO_ATTR_ID.keys():
            out_dct['per_size'][_size] = dict()
            for metric in ['precision', 'recall', 'iou']:
                out_dct['per_size'][_size][metric] = np.mean([vispr.overall_stats[metric][attr_id_to_idx[attr_id]]
                                                              for attr_id in SIZE_TO_ATTR_ID[_size]])

        # --- Per Mode
        out_dct['per_mode'] = dict()
        for _mode in MODE_TO_ATTR_ID.keys():
            out_dct['per_mode'][_mode] = dict()
            for metric in ['precision', 'recall', 'iou']:
                out_dct['per_mode'][_mode][metric] = np.mean([vispr.overall_stats[metric][attr_id_to_idx[attr_id]]
                                                              for attr_id in MODE_TO_ATTR_ID[_mode]])

        # --- Per Attribute
        out_dct['per_attribute'] = dict()
        for attr_idx, attr_id in enumerate(vispr.params.attrIds):
            out_dct['per_attribute'][attr_id] = dict()
            for metric in ['precision', 'recall', 'iou']:
                out_dct['per_attribute'][attr_id][metric] = vispr.overall_stats[metric][attr_idx]
        json.dump(out_dct, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    main()