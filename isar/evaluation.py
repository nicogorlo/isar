import numpy as np
import cv2

import os


class Evaluation():
    def __init__(self, datadir, eval_dir, dataset_name='DAVIS') -> None:
        self.datadir = datadir
        self.eval_dir = eval_dir
        self.dataset_name = dataset_name

        self.mIoU = 0

    def compute_IoU(self, seg1, seg2):
        intersection = np.logical_and(seg1, seg2)
        union = np.logical_or(seg1, seg2)
        iou_score = np.sum(intersection) / np.sum(union)

        if self.mIoU == 0:
            self.mIoU = iou_score
        else:
            self.mIoU = (self.mIoU + iou_score) / 2
        return iou_score
    
    def compute_accuracy(self, seg1, seg2):
        pass
    

    def visualize_gt_mask(self, img, seg_gt):
        dst = cv2.addWeighted(img.astype('uint8'), 0.7, seg_gt.astype('uint8'), 0.3, 0).astype('uint8')
        cv2.imshow('gt_mask', dst)


    
    def get_gt_mask(self, image_name):
        if self.dataset_name == 'DAVIS':
            gt_mask = cv2.imread(os.path.join(self.datadir, self.eval_dir, image_name.replace('jpg', 'png')))
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            gt_mask = gt_mask > 0
            return gt_mask
        else:
            raise NotImplementedError
