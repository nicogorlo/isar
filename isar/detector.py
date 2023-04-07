import requests
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import cv2

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

import itertools
import seaborn as sns

import PIL
from PIL import Image

import panopticapi
from panopticapi.utils import id2rgb, rgb2id

from reidentification import Reidentification

from time import time

from util.isar_utils import performance_measure

from owdetr_img_inference import OW_DETR

from segment_anything import sam_model_registry, SamPredictor

from segmentor import SegmentorSAM, SegmentorU2
         


class Detector():
    def __init__(self):
        # self.segmentor = SegmentorU2()
        self.segmentor = SegmentorSAM()
        self.reid = Reidentification()
        self.ow_detr = OW_DETR(keep_logits=0.7, keep_objectness=0.15)

        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.prob = None
        self.boxes = None

        self.min_similarity = 0.87

        self.cutout_features = None

        self.start_reid = False


    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self,x: torch.tensor) -> torch.tensor:
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox: torch.tensor, size: tuple) -> torch.tensor:
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def detect(self, image: np.ndarray):
        scores, boxes, labels, keep = self.ow_detr(image)

        self.prob = scores[0, keep]
        self.boxes = boxes[0, keep]
        if self.start_reid:
            match = self.get_most_similar_box(image)

            if match is not None:
                seg = self.segmentor(image, match.cpu().numpy().astype(int))
                cv2.imshow('seg', seg)
                cv2.imshow('match', self.get_cutout(image, match))
            else: 
                print("no match found")

        return scores[0, keep], boxes[0, keep]

    
    def contains(self, prob: torch.tensor, boxes: torch.tensor, x: int, y: int):
        contains = torch.argwhere((boxes[:,0] < x) & (x < boxes[:,2]) & (boxes[:,1] < y) & (y < boxes[:,3]))
        selected_boxes = boxes[contains][:,0]
        selected_probs = prob[contains][:,0]
        # best = selected_probs.max(-1).values.argmax(keepdim=True)
        return selected_probs[0].unsqueeze(0), selected_boxes[0].unsqueeze(0)
    
    
    def get_cutout(self, img: np.ndarray, bbox: torch.tensor) -> np.ndarray:
        bbox = bbox.cpu().numpy().astype(int)
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    

    def on_click(self, x: int, y: int, img: np.ndarray):
        print('selected coordinates - (x: {}, y: {})'.format(x, y))
        selected_prob, selected_box = self.contains(self.prob, self.boxes, x, y)

        cutout = self.get_cutout(img, selected_box[0])

        self.start_reid = True

        with performance_measure("CLIP feature extraction"):
            self.cutout_features = self.reid.extract_img_feat(cutout)

        seg = self.segmentor(img, selected_box[0].cpu().numpy().astype(int))

        freeze = img

        return cutout, seg, freeze, selected_prob, selected_box
    

    def get_most_similar_box(self, img: np.ndarray):
        sim = []

        boxes = self.boxes

        keep = self.prob > 0.01

        for box in boxes[keep]:
            cutout = self.get_cutout(img, box)
            
            # print("cutout shape: ", cutout.shape)
            if cutout.shape[0] <= 0 or cutout.shape[1] <= 0:
                sim.append(0)
                continue

            new_cutout_features = self.reid.extract_img_feat(cutout)
            sim.append(self.reid.get_similarity(new_cutout_features, self.cutout_features))
        
        max_sim = max(sim)
        if max_sim < self.min_similarity:
            return None
        else:
            return boxes[torch.argmax(torch.tensor(sim))]
