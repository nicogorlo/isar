import numpy as np
import cv2
import os

import torch
from torch import nn
import torchvision.transforms as T
torch.set_grad_enabled(False)

from reidentification import Reidentification

from util.isar_utils import performance_measure

from owdetr_img_inference import OW_DETR

from segmentor import SegmentorSAM

from params import OUTDIR, KEEP_LOGITS, KEEP_OBJECTNESS, ADA_THRESH_MAX_OVERLAP, ADA_THRESH_MULTIPLIER


class Detector():
    def __init__(self, device = "cpu", sam_model_type = "vit_h", use_precomputed_embeddings = False):

        self.segmentor = SegmentorSAM(device, sam_model_type)
        self.segmentor.use_precomputed_embedding = use_precomputed_embeddings

        self.reid = Reidentification()
        self.ow_detr = OW_DETR(keep_logits=KEEP_LOGITS, keep_objectness=KEEP_OBJECTNESS)

        self.prob: torch.Tensor = None
        self.boxes: torch.Tensor = None

        self.outdir = OUTDIR

        self.cutout_features = None
    
        self.show_images = True

        self.start_reid = False

        self.object_center = None

        self.it = 0

        ## for verbose
        self.adaptive_threshold = 0.7
        self.fixed_threshold = 0.87
        self.max_sim = 0.0

    
    def detect(self, img: np.ndarray, image_name: str, embedding: str):
        with performance_measure("OW DETR boxes"):
            scores, boxes, labels, keep = self.ow_detr(img)

            #threshold boxes to image boundaries:
            torch.clamp(boxes[:,0], 0, img.shape[1], out=boxes[:,0])
            torch.clamp(boxes[:,1], 0, img.shape[0], out=boxes[:,1])
            torch.clamp(boxes[:,2], 0, img.shape[1], out=boxes[:,2])
            torch.clamp(boxes[:,3], 0, img.shape[0], out=boxes[:,3])

            self.prob = scores[0, keep]
            self.boxes = boxes[0, keep]

        seg = None

        if self.start_reid:
            with performance_measure("Get most similar box"):
                match = self.get_most_similar_box(img)

            if match is not None:
                seg = self.segmentor(img, match, embedding)
                dst = cv2.addWeighted(img.astype('uint8'), 0.7, seg.astype('uint8'), 0.3, 0).astype('uint8')

                if self.show_images: 
                    dst = self.visualize_scores(seg, dst, color=(0,255,0))
                    cv2.rectangle (dst, (match[0], match[1]), (match[2], match[3]), (0,255,0), 2)
                    cv2.imshow('seg', dst)
                    cv2.imshow('match', self.get_cutout(img, match))

            else: 
                print("no match found")
                seg = np.zeros(img.shape)
                dst = cv2.addWeighted(img.astype('uint8'), 0.7, seg.astype('uint8'), 0.3, 0).astype('uint8')

                if self.show_images:
                    dst = self.visualize_scores(seg, dst, color=(0,0,255))

            cv2.imwrite(os.path.join(self.outdir, image_name), dst)
            self.it+=1

        return scores[0, keep], boxes[0, keep], seg

    
    def contains(self, prob: torch.Tensor, boxes: torch.Tensor, x: int, y: int):
        contains = torch.argwhere((boxes[:,0] < x) & (x < boxes[:,2]) & (boxes[:,1] < y) & (y < boxes[:,3]))
        selected_boxes = boxes[contains][:,0]
        selected_probs = prob[contains][:,0]
        # best = selected_probs.max(-1).values.argmax(keepdim=True)
        return selected_probs[0].unsqueeze(0), selected_boxes[0].unsqueeze(0)
    
    
    def get_cutout(self, img: np.ndarray, bbox) -> np.ndarray:
        if bbox.__class__ == torch.Tensor:
            bbox = bbox.cpu().numpy().astype(int)
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    

    def on_click(self, x: int, y: int, img: np.ndarray, embedding: str):
        print('selected coordinates - (x: {}, y: {})'.format(x, y))
        selected_prob, selected_box = self.contains(self.prob, self.boxes, x, y)

        cutout = self.get_cutout(img, selected_box[0])

        self.start_reid = True

        with performance_measure("CLIP feature extraction"):
            self.cutout_features = self.reid.extract_img_feat(cutout)

        seg = self.segmentor(img, selected_box[0].cpu().numpy().astype(int), embedding)

        freeze = img

        return cutout, seg, freeze, selected_prob, selected_box
    
    def on_button_press(self):
        self.start_reid = not self.start_reid

        if self.start_reid:
            print("start reid")
        else:
            print("stop reid")

    
    def compute_object_center(self, seg):
        """
        compute the center of the object
        """
        pass

    def get_most_similar_box(self, img: np.ndarray):
        sims = []

        boxes = self.boxes
        
        box_sims = []
        keep = self.prob > 0.01

        for idx, box in enumerate(boxes[keep]):
            cutout = self.get_cutout(img, box)
            # print("cutout shape: ", cutout.shape)
            if cutout.shape[0] <= 0 or cutout.shape[1] <= 0:
                sims.append(0)
                box_sims.append({"box": box, "similarity": 0})
                continue

            new_cutout_features = self.reid.extract_img_feat(cutout)
            similarity = self.reid.get_similarity(new_cutout_features, self.cutout_features)
            sims.append(similarity)
            entry = {"box": box.cpu().numpy().astype(int), "similarity": similarity}
            box_sims.append(entry)
        
        box_sims.sort(key=lambda x: x["similarity"], reverse=True)
        max_sim = box_sims[0]["similarity"]
        color = [(0, int(255*x), int(255*(1-x))) for x in [i/len(box_sims) for i in range(len(box_sims))]]
        print("new matching:")
        print("number of boxes: ", len(box_sims))
        print("max similarity: ", max_sim)
        i = 0
        self.max_sim = max_sim
        boundary = min(5, len(box_sims))
        meansim = box_sims[1]["similarity"]
        count = 0
        sum = 0
        while i < boundary:
            if self.calculate_IoSmallerBox(torch.tensor(box_sims[i]["box"]), torch.tensor(box_sims[0]["box"])) > ADA_THRESH_MAX_OVERLAP:
                boundary = min(boundary+1, len(box_sims))
                i+=1
                continue
            # cv2.rectangle(img, (box_sims[i]["box"][0], box_sims[i]["box"][1]), (box_sims[i]["box"][2], box_sims[i]["box"][3]), color[i], 2)
            # cv2.putText(img, str(box_sims[i]["similarity"]), (box_sims[i]["box"][0], box_sims[i]["box"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[i], 1, cv2.LINE_AA)
            count += 1
            sum += box_sims[i]["similarity"]
            # print("similarity: ", box_sims[i]["similarity"], "IoU with chosen box: ", self.calculate_iou(torch.tensor(box_sims[i]["box"]), torch.tensor(box_sims[0]["box"])))
            i+=1
        if count == 0:
            meansim = 0.5
        else:
            meansim = sum / count
        # print("mean similarity of 5 next best: ", meansim)
        # print("threshold: ", meansim * ADA_THRESH_MULTIPLIER)
        # self.adaptive_threshold = meansim * ADA_THRESH_MULTIPLIER
        if max_sim < self.fixed_threshold: # self.adaptive_threshold:
            # cv2.rectangle(img, (box_sims[0]["box"][0], box_sims[0]["box"][1]), (box_sims[0]["box"][2], box_sims[0]["box"][3]), (0,0,255), 2)
            return None
        else:
            return box_sims[0]["box"]


    def calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor):

        x_left = torch.max(box1[0], box2[0])
        y_top = torch.max(box1[1], box2[1])
        x_right = torch.min(box1[2], box2[2])
        y_bottom = torch.min(box1[3], box2[3])

        intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)

        # Calculate the union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU and return
        iou = intersection_area / union_area
        return iou.item()
    

    def calculate_IoSmallerBox(self, box1: torch.Tensor, box2: torch.Tensor):
            
            x_left = torch.max(box1[0], box2[0])
            y_top = torch.max(box1[1], box2[1])
            x_right = torch.min(box1[2], box2[2])
            y_bottom = torch.min(box1[3], box2[3])
    
            intersection_area = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
    
            # Calculate the smaller box area
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            smallerBoxArea = min(box1_area, box2_area)
    
            # Calculate IoU and return
            iosb = intersection_area / smallerBoxArea
            return iosb.item()
    
    def visualize_scores(self, seg, dst, color):
        cv2.putText(dst, f"similarity:", (int(seg.shape[1]*0.7), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(dst, f"{round(self.max_sim, 4)}", (int(seg.shape[1]*0.90), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(dst, f"adaptive threshold:", (int(seg.shape[1]*0.7), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(dst, f"{round(self.adaptive_threshold,4)}", (int(seg.shape[1]*0.90), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return dst