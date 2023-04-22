import numpy as np
import cv2

import os


class Evaluation():
    def __init__(self, eval_dir) -> None:
        self.eval_dir = eval_dir

        self.ious = {}

        self.mIoU = 0

        self.misclassifications = 0
        self.false_detection_visible = 0
        self.total_frames_visible = 0

        self.false_detection_not_visible = 0
        self.total_frames_not_visible = 0

    def compute_IoU(self, seg1, seg2):
        intersection = np.logical_and(seg1, seg2)
        union = np.logical_or(seg1, seg2)
        iou_score = np.sum(intersection) / np.sum(union)

        return iou_score
    

    def visualize_gt_mask(self, img, seg_gt):
        dst = cv2.addWeighted(img.astype('uint8'), 0.7, seg_gt.astype('uint8'), 0.3, 0).astype('uint8')
        cv2.imshow('gt_mask', dst)

    
    def get_gt_mask(self, image_name):
        gt_mask = cv2.imread(os.path.join(self.eval_dir, image_name))
        if gt_mask is None:
            gt_mask = cv2.imread(os.path.join(self.eval_dir, image_name.replace('jpg', 'png')))
        if gt_mask is None:
            raise Exception("No ground truth mask found for image {}".format(image_name))
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        gt_mask = gt_mask > 0
        return gt_mask
    


    #################################################
    ###########  Evaluation Metric Ideas  ###########
    #################################################

    def object_lost_counter(self, IoU_score):
        if IoU_score < 0.4:
            self.misclassifications += 1
            print("IoU < 0.4")


    def false_detection_rate_visible_counter(self, mask, gt_mask, IoU_score):
        if IoU_score < 0.1 and np.sum(mask) > np.sum(gt_mask) * IoU_score:
            self.false_detection_visible += 1
            # print("object lost")

    
    def false_detection_rate_not_visible_counter(self, mask, gt_mask):
        if sum(gt_mask) == 0 and np.sum(mask) > 0:
            self.false_detection_not_visible += 1
            # print("object lost")

    def object_visible_gt_counter(self, gt_mask):
        if sum(gt_mask) == 0:
            self.total_frames_not_visible += 1
            # print("object not visible")
        else:
            self.total_frames_visible += 1
            # print("object visible")
    

    def compute_evaluation_metrics(self, mask, gt_mask, image_name):
        iou = self.compute_IoU(mask, gt_mask)
        self.ious[image_name] = iou
        self.object_lost_counter(iou)
        self.false_detection_rate_visible_counter(mask, gt_mask, iou)
        self.false_detection_rate_not_visible_counter(mask, gt_mask)
        self.object_visible_gt_counter(gt_mask)

    def report_results(self, scene_name):
        self.mIoU = np.mean(self.ious.values())
        print("scene mIoU: {}".format(self.mIoU))
        print("misclassifications: {}".format(self.misclassifications))
        print("false detection ratio visible: {}".format(self.false_detection_visible / self.total_frames_visible))
        print("false detection ratio not visible: {}".format(self.false_detection_not_visible / self.total_frames_not_visible))

        return {scene_name: {'mIoU': self.mIoU, 'misclassifications': self.misclassifications, 'false_detection_ratio_visible': self.false_detection_visible / self.total_frames_visible, 'false_detection_ratio_not_visible': self.false_detection_not_visible / self.total_frames_not_visible}}