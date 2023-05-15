import numpy as np
import cv2

from sklearn.metrics import precision_recall_fscore_support

import os


class Evaluation():
    def __init__(self, eval_dir) -> None:
        self.eval_dir = eval_dir

        self.ious = {}
        self.bound_f_measures = {}

        self.mIoU = 0
        self.mBoundF = 0

        self.misclassifications = 0
        self.false_detection_visible = 0
        self.total_frames_visible = 0

        self.false_detection_not_visible = 0
        self.total_frames_not_visible = 0

    def compute_IoU(self, seg1, seg2):
        "computes the Jaccard index (intersection over Union) of two binary masks"
        intersection = np.logical_and(seg1, seg2)
        union = np.logical_or(seg1, seg2)
        if np.sum(union) == 0:
            return 1.0
        else: 
            iou_score = np.sum(intersection) / np.sum(union)

        return iou_score
    

    def visualize_gt_mask(self, img, seg_gt):
        dst = cv2.addWeighted(img.astype('uint8'), 0.7, seg_gt.astype('uint8'), 0.3, 0).astype('uint8')
        cv2.imshow('gt_mask', dst)

    
    def get_gt_mask(self, image_name):
        if os.path.exists(os.path.join(self.eval_dir, image_name)):
            gt_mask = cv2.imread(os.path.join(self.eval_dir, image_name))
        elif os.path.exists(os.path.join(self.eval_dir, image_name.replace('jpg', 'png'))):
            gt_mask = cv2.imread(os.path.join(self.eval_dir, image_name.replace('jpg', 'png')))
        else:
            raise Exception("No ground truth mask found for image {}".format(image_name))
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        if gt_mask.sum() > 1.0:
            gt_mask = cv2.GaussianBlur(gt_mask, (21, 21), 0)
            gt_mask = (gt_mask > gt_mask.max()/2)
        else: 
            gt_mask = (gt_mask >= 0.5)
        return gt_mask.astype("uint8")


    #################################################
    ###########  Evaluation Metric Ideas  ###########
    #################################################


    def misclassification_counter(self, IoU_score):
        if IoU_score < 0.4:
            self.misclassifications += 1
            print("IoU < 0.4")


    def false_detection_rate_visible_counter(self, mask, gt_mask, IoU_score):
        if IoU_score < 0.1 and np.sum(mask) > np.sum(gt_mask) * IoU_score and np.sum(gt_mask) > 0:
            self.false_detection_visible += 1
            # print("object lost")

    
    def false_detection_rate_not_visible_counter(self, mask, gt_mask):
        if np.sum(gt_mask) == 0 and np.sum(mask) > 0:
            self.false_detection_not_visible += 1
            # print("object lost")

    def object_visible_gt_counter(self, gt_mask):
        if np.sum(gt_mask) == 0:
            self.total_frames_not_visible += 1
            # print("object not visible")
        else:
            self.total_frames_visible += 1
            # print("object visible")
    

    def find_contours(self, mask):
        
        # Use OpenCV to find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Create an empty image to draw the contours
        contour_image = np.zeros_like(mask)
        cv2.drawContours(contour_image, contours, -1, (255), 1)
        
        return contour_image

    def compute_boundary_f_measure(self, predicted_mask, gt_mask):
        # Find contours
        predicted_contours = self.find_contours(predicted_mask)
        gt_contours = self.find_contours(gt_mask)
        
        # Flatten and binarize
        predicted_contours = (predicted_contours.flatten() > 0).astype(int)
        gt_contours = (gt_contours.flatten() > 0).astype(int)
        
        # Calculate precision, recall, and F-measure TODO: compute separately to take care of zero case.
        precision, recall, f_measure, _ = precision_recall_fscore_support(gt_contours, predicted_contours, average='binary', zero_division=1)
        
        return precision, recall, f_measure
        

    def compute_evaluation_metrics(self, mask, gt_mask, image_name):
        iou = self.compute_IoU(mask, gt_mask)
        prec, rec, boundary_f_measure = self.compute_boundary_f_measure(mask.astype("uint8"), gt_mask)
        self.ious[image_name] = iou
        self.bound_f_measures[image_name] = boundary_f_measure
        self.misclassification_counter(iou)
        self.false_detection_rate_visible_counter(mask, gt_mask, iou)
        self.false_detection_rate_not_visible_counter(mask, gt_mask)
        self.object_visible_gt_counter(gt_mask)

    def report_results(self, scene_name):
        self.mIoU = np.mean(list(self.ious.values()))
        self.mBoundF = np.mean(list(self.bound_f_measures.values()))
        false_detection_rate_visible = self.false_detection_visible / self.total_frames_visible
        if self.total_frames_not_visible == 0:
            false_detection_rate_not_visible = None
        else:
            false_detection_rate_not_visible = self.false_detection_not_visible / self.total_frames_not_visible
        print("scene mIoU: {}".format(self.mIoU))
        print("scene boundary f measure: {}".format(self.mBoundF))
        print("misclassification rate: {}".format(self.misclassifications/len(self.ious)))
        print("false detection ratio visible: {}".format(false_detection_rate_visible))
        print("false detection ratio not visible: {}".format(false_detection_rate_not_visible))

        return {scene_name: {'mIoU': self.mIoU, 'mBoundF': self.mBoundF, 'misclassification_rate': self.misclassifications/len(self.ious), 'false_detection_ratio_visible': false_detection_rate_visible, 'false_detection_ratio_not_visible': false_detection_rate_not_visible}}