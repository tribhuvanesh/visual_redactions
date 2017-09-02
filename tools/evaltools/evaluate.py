#!/usr/bin/python
"""Given a GT and a Prediction file, evaluate predictions
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
from scipy.misc import imread

import pprint

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class VISPRSegEval:
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
    ]
    """
    def __init__(self, gt_path, pred_path):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.vispr_gt_full = json.load(open(gt_path))
        self.vispr_gt = self.vispr_gt_full['annotations']
        self.vispr_pred = json.load(open(pred_path))

        self.evalImgs = dd(list)  # per-image per-category evaluation results [KxAxI] elements
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

        self.stats = []

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

        # --- Stats -------------------------------------------------------------------------------------
        print
        for idx, (low, high) in enumerate(self.params.areaRng):
            count = 0
            for gts in self._gts.values():
                for gt in gts:
                    if low < gt['area'] < high:
                        count += 1
            print '# GT objects ({}) = {}'.format(self.params.areaRngLbl[idx], count)
            count = 0
            for pds in self._pds.values():
                for pd in pds:
                    if low < pd['area'] < high:
                        count += 1
            print '# PD objects ({}) = {}'.format(self.params.areaRngLbl[idx], count)

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params

        p.imgIds = list(np.unique(p.imgIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self.prepare()
        # loop through images, area range, max detection number
        attr_ids = p.attrIds

        computeIoU = self.computeIoU
        self.ious = {(image_id, attr_id): computeIoU(image_id, attr_id)
                     for image_id in p.imgIds
                     for attr_id in attr_ids}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(image_id, attr_id, areaRng, maxDet)
                         for attr_id in attr_ids
                         for areaRng in p.areaRng
                         for image_id in p.imgIds
                         ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, image_id, attr_id):
        """
        If there are <n_g> GT annotations and <n_d> detections, this produces a IoU matrix of size <n_d x n_g>
        :param image_id:
        :param attr_id:
        :return:
        """
        p = self.params

        gt = self._gts[image_id, attr_id]  # List of annotations for this image-category
        dt = self._pds[image_id, attr_id]  # List of predictions for this image-category

        if len(gt) == 0 and len(dt) == 0:
            return []

        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        g = [g['segmentation'] for g in gt]
        d = [d['segmentation'] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = mask_utils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._pds[imgId, catId]
        else:
            gt = [_ for cId in p.attrIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.attrIds for _ in self._pds[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        ignore_count = 0
        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
                # print "g['ignore'] = {}, (g['area'](={}) < aRng[0](={}) or g['area'](={}) > aRng[1](={}))".format(g['ignore'], g['area'], aRng[0], g['area'], aRng[1])
                ignore_count += 1
            else:
                g['_ignore'] = 0
        # print '{} / {} ignored'.format(ignore_count, len(gt))

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.attrIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.attrIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarize_cats(ap=1, iouThr=0.5, areaRng='all', maxDets=100):
            p = self.params

            for k, attr_id in enumerate(self.params.attrIds):
                iStr = '{:<30} {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
                titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
                typeStr = '(AP)' if ap == 1 else '(AR)'
                iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                    if iouThr is None else '{:0.2f}'.format(iouThr)

                aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
                mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, k, aind, mind]
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s > -1])
                print(iStr.format(attr_id, titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')

        self.stats = _summarizeDets()
        print
        _summarize_cats(iouThr=0.5)

    def __str__(self):
        self.summarize()


class Params:
    """
    Adapted from coco evaluation api
    """
    def setDetParams(self):
        self.imgIds = []
        self.attrIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def __init__(self):
        self.setDetParams()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", type=str, help="GT File")
    parser.add_argument("pred_file", type=str, help="Predicted file")
    args = parser.parse_args()
    params = vars(args)
    vispr = VISPRSegEval(params['gt_file'], params['pred_file'])
    print
    vispr.evaluate()
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(vispr.evalImgs)
    vispr.accumulate()
    vispr.summarize()

if __name__ == '__main__':
    main()