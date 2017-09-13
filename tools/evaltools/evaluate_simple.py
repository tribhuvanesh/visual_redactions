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
le
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
        for pred in self.vispr_pred:
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

        precision = -np.ones((n_imgs, n_attr), dtype=np.float32)
        recall    = -np.ones((n_imgs, n_attr), dtype=np.float32)
        iou       = -np.ones((n_imgs, n_attr), dtype=np.float32)

        if visualize_dir and not osp.exists(visualize_dir):
            print 'Path {} does not exist. Creating it...'.format(visualize_dir)
            os.mkdir(visualize_dir)

        for image_idx, image_id in enumerate(p.imgIds):
            sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (image_idx, n_imgs,
                                                                    (image_idx * 100.0) / n_imgs))
            sys.stdout.flush()

            img_w, img_h = self.vispr_gt[image_id]['image_width'], self.vispr_gt[image_id]['image_height']

            if visualize_dir is not None:
                mask_stats_list = []  # (attr_id, gt_mask, pred_mask, prec, rec, iou)
            else:
                mask_stats_list = None

            for attr_idx, attr_id in enumerate(attr_ids):
                key = (image_id, attr_id)
                gt_exists = key in self._gts
                pd_exists = key in self._pds

                if (not gt_exists) and (not pd_exists):
                    continue

                # Create a GT bimask
                gt = self._gts[(image_id, attr_id)]
                # gt_mask, ig_mask = self.anno_to_bimask(gt, (img_w, img_h))
                try:
                    gt_mask, ig_mask = self.anno_to_bimask_org(gt, (img_w, img_h))
                except AssertionError:
                    print image_id
                    raise

                # Create Predicted bimask
                pd = self._pds[(image_id, attr_id)]
                # pd_mask, _ = self.anno_to_bimask(pd, (img_w, img_h))
                try:
                    pd_mask, _ = self.anno_to_bimask_org(pd, (img_w, img_h))
                except AssertionError:
                    print image_id
                    raise

                # FIXME Ignoring crowd masks for now
                _prec, _rec, _iou = compute_eval_metrics(gt_mask, pd_mask)

                precision[image_idx, attr_idx] = _prec
                recall[image_idx, attr_idx] = _rec
                iou[image_idx, attr_idx] = _iou

                if mask_stats_list is not None:
                    mask_stats_list.append((
                        attr_id,
                        gt_mask.copy(), pd_mask.copy(),
                        _prec, _rec, _iou
                    ))

                del gt_mask
                del pd_mask

            if mask_stats_list is not None:
                vis_out_path = osp.join(visualize_dir, image_id + '.jpg')
                self.visualize_img(mask_stats_list, image_id, vis_out_path)
                del mask_stats_list

        self.evalImgs['precision'] = precision
        self.evalImgs['recall'] = recall
        self.evalImgs['iou'] = iou

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def visualize_img(self, mask_stats_list, image_id, vis_out_path):
        """

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

        for _idx, (attr_id, gt_mask, pred_mask, _prec, _rec, _iou) in enumerate(mask_stats_list):
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
            text_str = "precision: {:.2f}\nrecall: {:.2f}\niou: {:.2f}".format(_prec*100, _rec*100, _iou*100)
            ax.text(0.1, 0.5, text_str, fontsize='large')

        plt.tight_layout()
        plt.savefig(vis_out_path, bbox_inches='tight')
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

        # Calculate per-class stats
        overall_precision = np.zeros(len(p.attrIds))
        overall_recall = np.zeros(len(p.attrIds))
        overall_iou = np.zeros(len(p.attrIds))

        for attr_idx, attr_id in enumerate(p.attrIds):
            m = precision[:, attr_idx]
            overall_precision[attr_idx] = np.mean(m[m > -1])
            m = recall[:, attr_idx]
            overall_recall[attr_idx] = np.mean(m[m > -1])
            m = iou[:, attr_idx]
            overall_iou[attr_idx] = np.mean(m[m > -1])

        self.overall_stats['precision'] = overall_precision
        self.overall_stats['recall'] = overall_recall
        self.overall_stats['iou'] = overall_iou

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
    args = parser.parse_args()
    params = vars(args)
    vispr = VISPRSegEvalSimple(params['gt_file'], params['pred_file'])
    print
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

        out_dct = dict()
        out_dct['overall'] = {
            'precision': np.mean(vispr.overall_stats['precision']),
            'recall': np.mean(vispr.overall_stats['recall']),
            'iou': np.mean(vispr.overall_stats['iou']),
        }
        out_dct['per_attribute'] = dict()
        for attr_idx, attr_id in enumerate(vispr.params.attrIds):
            out_dct['per_attribute'][attr_id] = dict()
            for metric in ['precision', 'recall', 'iou']:
                out_dct['per_attribute'][attr_id][metric] = vispr.overall_stats[metric][attr_idx]
        json.dump(out_dct, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    main()