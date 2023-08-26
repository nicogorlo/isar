import numpy as np
import cv2

from sklearn.metrics import precision_recall_fscore_support

import os


class Evaluation():
    def __init__(self, eval_dir, info = None) -> None:
        self.eval_dir = eval_dir

        self.ious = {}
        self.bound_f_measures = {}

        self.mIoU = {}
        self.mBoundF = {}

        self.misclassifications = {}
        self.false_detection_visible = {}
        self.total_frames_visible = {}

        self.false_detection_not_visible = {}
        self.total_frames_not_visible = {}

        if info is not None:
            self.info = info
            self.semantic_ids = info['semantic_ids']
            self.color_map = info['color_map']

            for id in self.semantic_ids:
                self.ious[id] = {}
                self.bound_f_measures[id] = {}
                self.misclassifications[id] = 0
                self.false_detection_visible[id] = 0
                self.total_frames_visible[id] = 0
                self.false_detection_not_visible[id] = 0
                self.total_frames_not_visible[id] = 0
                self.mIoU[id] = 0
                self.mBoundF[id] = 0

        else:
            raise Exception("No info.json file found in {}".format(eval_dir.split('/')[-4]))

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
        if os.path.exists(os.path.join(self.eval_dir, image_name.replace("jpg", "npy"))):
            gt_mask = np.load(os.path.join(self.eval_dir, image_name.replace("jpg", "npy")))
        elif os.path.exists(os.path.join(self.eval_dir, image_name.replace("jpg", "png"))):
            gt_mask = cv2.imread(os.path.join(self.eval_dir, image_name.replace("jpg", "png")))
        else:
            raise Exception("No ground truth mask found for image {}".format(image_name))
        return gt_mask


    def misclassification_counter(self, id, IoU_score):
        if IoU_score < 0.4:
            self.misclassifications[id] += 1
            print("IoU < 0.4")


    def false_detection_rate_visible_counter(self, id, mask, gt_mask, IoU_score):
        if IoU_score < 0.1 and np.sum(mask) > np.sum(gt_mask) * IoU_score and np.sum(gt_mask) > 0:
            self.false_detection_visible[id] += 1

    
    def false_detection_rate_not_visible_counter(self, id, mask, gt_mask):
        if np.sum(gt_mask) == 0 and np.sum(mask) > 0:
            self.false_detection_not_visible[id] += 1

    def object_visible_gt_counter(self, id, gt_mask):
        if np.sum(gt_mask) == 0:
            self.total_frames_not_visible[id] += 1
        else:
            self.total_frames_visible[id] += 1
    

    def find_contours(self, mask):
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        contour_image = np.zeros_like(mask)
        cv2.drawContours(contour_image, contours, -1, (255), 1)
        
        return contour_image

    def compute_boundary_f_measure(self, predicted_mask, gt_mask):
        if predicted_mask.sum() > 0:
            predicted_contours = self.find_contours(predicted_mask)
            gt_contours = self.find_contours(gt_mask)
            
            predicted_contours = (predicted_contours.flatten() > 0).astype(int)
            gt_contours = (gt_contours.flatten() > 0).astype(int)
            
            precision, recall, f_measure, _ = precision_recall_fscore_support(gt_contours, predicted_contours, average='binary', zero_division=1)
            
            return precision, recall, f_measure
        
        else:
            return 0.0, 0.0, 0.0
        

    def compute_evaluation_metrics(self, mask, image_name):
        gt_mask = self.get_gt_mask(image_name)
        for id in self.semantic_ids:
            color = np.array(self.color_map[str(id)])

            mask_bin = np.all(mask==color, axis=-1)
            gt_mask_bin = (gt_mask == id)

            if gt_mask_bin.sum() > 0:
                iou = self.compute_IoU(mask_bin, gt_mask_bin)
                prec, rec, boundary_f_measure = self.compute_boundary_f_measure(mask_bin.astype("uint8"), gt_mask_bin.astype("uint8"))
                self.ious[id][image_name] = iou
                self.bound_f_measures[id][image_name] = boundary_f_measure
                self.misclassification_counter(id, iou)
                self.false_detection_rate_visible_counter(id, mask_bin, gt_mask_bin, iou)
            self.false_detection_rate_not_visible_counter(id, mask_bin, gt_mask_bin)
            self.object_visible_gt_counter(id, gt_mask_bin)

    def report_results(self, scene_name):
        false_detection_rate_visible = {}
        false_detection_rate_not_visible = {}
        misclassification_rate = {}
        for id in self.semantic_ids:
            self.mIoU[id] = np.mean(list(self.ious[id].values()))
            self.mBoundF[id] = np.mean(list(self.bound_f_measures[id].values()))
            if self.total_frames_visible[id] == 0:
                false_detection_rate_visible[id] = None
            else:
                false_detection_rate_visible[id] = self.false_detection_visible[id] / self.total_frames_visible[id]
            if self.total_frames_not_visible[id] == 0:
                false_detection_rate_not_visible[id] = None
            else:
                false_detection_rate_not_visible[id] = self.false_detection_not_visible[id] / self.total_frames_not_visible[id]
            
            if len(self.ious[id]) > 0: 
                misclassification_rate[id] = self.misclassifications[id]/len(self.ious[id])

        scene_miou = np.mean(list(self.mIoU.values()))
        scene_mboundf = np.mean(list(self.mBoundF.values()))
        scene_misclassification_rate = np.mean(list(misclassification_rate.values()))

        print("scene mIoU: {}".format(scene_miou))
        print("scene boundary f measure: {}".format(scene_mboundf))
        print("scene misclassification rate: {}".format(scene_misclassification_rate))
        print("false detection ratio visible: {}".format(false_detection_rate_visible))
        print("false detection ratio not visible: {}".format(false_detection_rate_not_visible))

        return {
                scene_name: 
                    {
                    'mIoU': scene_miou, 
                    'mBoundF': scene_mboundf, 
                    'misclassification_rate': scene_misclassification_rate, 
                    'false_detection_ratio_visible': false_detection_rate_visible, 
                    'false_detection_ratio_not_visible': false_detection_rate_not_visible,
                    'iou_detailed': self.ious
                    }
                    }