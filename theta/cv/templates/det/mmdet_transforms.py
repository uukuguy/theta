import inspect

import numpy as np
from numpy import random

import mmcv
from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from mmdet.datasets.builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    #  from albumentations import albumentations
    #  from albumentations.albumentations import Compose
    from . import albumentations
    from .albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

#  @PIPELINES.register_module
#  class CopyPaste(object):
#      def __init__(self, p=0.3, lambd=0.5, categry_id=-1):
#          self.lambd = lambd
#          self.p = p
#          self.category_id = category_id
#          self.img2 = None
#          self.boxes2 = None
#          self.labels2 = None
#
#      def __call__(self, results):
#          img1, boxes1, labels1 = [
#              results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
#          ]
#          if random.random() < self.p and self.img2 is not None and img1.shape[
#                  1] == self.img2.shape[1]:
#
#              height = max(img1.shape[0], self.img2.shape[0])
#              width = max(img1.shape[1], self.img2.shape[1])
#
#              #  mixup_image = np.zeros([height, width, 3], dtype='float32')
#              #  mixup_image[:img1.shape[0], :img1.
#              #              shape[1], :] = img1.astype('float32') * self.lambd
#              #  mixup_image[:self.img2.shape[0], :self.img2.
#              #              shape[1], :] += self.img2.astype('float32') * (
#              #                  1. - self.lambd)
#              #  mixup_image = mixup_image.astype('uint8')
#
#              #  mixup_boxes = np.vstack((boxes1, self.boxes2))
#              #  mixup_label = np.hstack((labels1, self.labels2))
#              #
#              #  results['img'] = mixup_image
#              #  results['gt_bboxes'] = mixup_boxes
#              #  results['gt_labels'] = mixup_label
#
#              standby_bboxes = []
#              standby_labels = []
#              for ix, label in enumerate(self.labels2):
#                  if label == self.category_id:
#                      standby_bboxes.append(self.boxes2[ix])
#                      standby_labels.append(self.lables2[ix])
#              if standby_bboxes:
#                  cp_boxes = np.vstack((boxes1, standby_bboxes))
#                  cp_labels = np.hstack((labels1, standby_labels))
#              else:
#                  cp_boxes = boxes1
#                  cp_labels = labels1
#
#              results['img'] = mixup_image
#              results['gt_bboxes'] = mixup_boxes
#              results['gt_labels'] = mixup_label
#          else:
#              pass
#          self.img2 = img1
#          self.boxes2 = boxes1
#          self.labels2 = labels1
#          return results
#


@PIPELINES.register_module
class MixUp(object):
    def __init__(self, p=0.3, lambd=0.5):
        self.lambd = lambd
        self.p = p
        self.img2 = None
        self.boxes2 = None
        self.labels2 = None

    def __call__(self, results):
        img1, boxes1, labels1 = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        if random.random() < self.p and self.img2 is not None and img1.shape[
                1] == self.img2.shape[1]:

            height = max(img1.shape[0], self.img2.shape[0])
            width = max(img1.shape[1], self.img2.shape[1])
            mixup_image = np.zeros([height, width, 3], dtype='float32')
            mixup_image[:img1.shape[0], :img1.
                        shape[1], :] = img1.astype('float32') * self.lambd
            mixup_image[:self.img2.shape[0], :self.img2.
                        shape[1], :] += self.img2.astype('float32') * (
                            1. - self.lambd)
            mixup_image = mixup_image.astype('uint8')
            mixup_boxes = np.vstack((boxes1, self.boxes2))
            mixup_label = np.hstack((labels1, self.labels2))
            results['img'] = mixup_image
            results['gt_bboxes'] = mixup_boxes
            results['gt_labels'] = mixup_label
        else:
            pass
        self.img2 = img1
        self.boxes2 = boxes1
        self.labels2 = labels1
        return results


@PIPELINES.register_module
class RandomVFlip(object):
    """Flip the image & bbox & mask.
    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    Args:
        flip_ratio (float, optional): The flipping probability.
    """
    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert 0 <= flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        h = img_shape[0]
        flipped = bboxes.copy()
        flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
        flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        return flipped

    def __call__(self, results):
        if 'vflip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['vflip'] = flip
        if results['vflip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'], direction="vertical")
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class BBoxJitter(object):
    """
    bbox jitter
    Args:
        min (int, optional): min scale
        max (int, optional): max scale
        ## origin w scale
    """
    def __init__(self, min=0, max=2):
        self.min_scale = min
        self.max_scale = max
        self.count = 0

    def bbox_jitter(self, bboxes, img_shape):
        """Flip bboxes horizontally.
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        if len(bboxes) == 0:
            return bboxes
        jitter_bboxes = []
        for box in bboxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            scale = np.random.uniform(self.min_scale, self.max_scale)
            w = w * scale / 2.
            h = h * scale / 2.
            xmin = center_x - w
            ymin = center_y - h
            xmax = center_x + w
            ymax = center_y + h
            box2 = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            jitter_bboxes.append(box2)
        jitter_bboxes = np.array(jitter_bboxes, dtype=np.float32)
        jitter_bboxes[:, 0::2] = np.clip(jitter_bboxes[:, 0::2], 0,
                                         img_shape[1] - 1)
        jitter_bboxes[:, 1::2] = np.clip(jitter_bboxes[:, 1::2], 0,
                                         img_shape[0] - 1)
        return jitter_bboxes

    def __call__(self, results):
        for key in results.get('bbox_fields', []):
            results[key] = self.bbox_jitter(results[key], results['img_shape'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_jitter={}-{})'.format(
            self.min_scale, self.max_scale)
