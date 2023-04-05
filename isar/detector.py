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

class Detector():
    def __init__(self):
        self.segmentor = Segmentor()
        self.reid = Reidentification()
        self.ow_detr = OW_DETR(keep_logits=0.7, keep_objectness=0.15)

        # self.detr = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        # self.detr.eval()

        # self.detr_seg, self.postprocessor_seg = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, num_classes=250, return_postprocessor=True)
        # self.detr_seg.eval()

        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.prob = None
        self.boxes = None

        self.cutout_features = None

        self.start_reid = False

        # COCO classes, not used in OW approach
        self.CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        # colors for visualization
        self.COLORS = [[123, 10, 15], [199, 145, 1], [200, 30, 0],
                [0, 120, 190], [0, 15, 123], [10, 140, 20]]

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
    
    # def detect(self, image: np.ndarray):
    #     # mean-std normalize the input image (batch-size: 1)
    #     im = Image.fromarray(image.astype('uint8'), 'RGB')
    #     img = self.transform(im).unsqueeze(0)

    #     # propagate through the model
    #     outputs = self.detr(img)

    #     # keep all predictions: (with 0.0+ confidence which is all, default: 0.7)
    #     probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    #     keep = probas.max(-1).values > 0.0

    #     # convert boxes from [0; 1] to image scales
    #     bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    #     self.prob = probas[keep]
    #     self.boxes = bboxes_scaled

    #     if self.start_reid:
    #         match = self.get_most_similar_box(image)
    #         if match is not None:
    #             cv2.imshow('match', self.get_cutout(image, match))
    #         else: 
    #             print("no match found")


        # return probas[keep], bboxes_scaled
    
    def detect(self, image: np.ndarray):
        scores, boxes, labels, keep = self.ow_detr(image)

        self.prob = scores[0, keep]
        self.boxes = boxes[0, keep]
        if self.start_reid:
            match = self.get_most_similar_box(image)
            if match is not None:
                cv2.imshow('match', self.get_cutout(image, match))
            else: 
                print("no match found")

        return scores[0, keep], boxes[0, keep]
        
    def get_seg_from_cutout(self, cutout):
        seg = self.segmentor(cutout)

        return seg
    
    
    def contains(self, prob: torch.tensor, boxes: torch.tensor, x: int, y: int):
        contains = torch.argwhere((boxes[:,0] < x) & (x < boxes[:,2]) & (boxes[:,1] < y) & (y < boxes[:,3]))
        selected_boxes = boxes[contains][:,0]
        selected_probs = prob[contains][:,0]
        # best = selected_probs.max(-1).values.argmax(keepdim=True)
        return selected_probs[0].unsqueeze(0), selected_boxes[0].unsqueeze(0)
    
    
    def get_cutout(self, img: np.ndarray, bbox: torch.tensor) -> np.ndarray:
        bbox = bbox.cpu().numpy()
        bbox = bbox.astype(int)
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    

    def on_click(self, x: int, y: int, img: np.ndarray):
        print('selected coordinates - (x: {}, y: {})'.format(x, y))
        selected_prob, selected_box = self.contains(self.prob, self.boxes, x, y)
        # cutout = self.get_cutout(img, selected_box[0])
        cutout = self.get_cutout(img, selected_box[0])

        self.start_reid = True

        with performance_measure("CLIP feature extraction"):
            self.cutout_features = self.reid.extract_img_feat(cutout)

        # sim = self.reid.get_similarity(self.cutout_features, self.reid.chair_feature)

        # print('similarity: ', sim)

        seg = self.get_seg_from_cutout(cutout)

        freeze = img

        return cutout, seg, freeze, selected_prob, selected_box
    

    def get_most_similar_box(self, img: np.ndarray):
        sim = []

        boxes = self.boxes

        keep = self.prob > 0.01   # essentially does nothing, however even just increasign to 0.001 
                                                    # will cause the program to not function as intended    
        # print("boxes shape: ", boxes.shape)
        # print("retained boxes shape: ", boxes[keep].shape)

        for box in boxes[keep]:
            cutout = self.get_cutout(img, box)
            
            # print("cutout shape: ", cutout.shape)
            if cutout.shape[0] <= 0 or cutout.shape[1] <= 0:
                sim.append(0)
                continue

            new_cutout_features = self.reid.extract_img_feat(cutout)
            sim.append(self.reid.get_similarity(new_cutout_features, self.cutout_features))
        
        max_sim = max(sim)
        if max_sim < 0.87:
            return None
        else:
            return boxes[torch.argmax(torch.tensor(sim))]



from external.u2net.model import U2NET # full size version 173.6 MB
from external.u2net.data_loader import RescaleT
from external.u2net.data_loader import ToTensorLab
from external.u2net.data_loader import ToTensor
from external.u2net.data_loader import SalObjDataset
from external.u2net.data_loader import DataLoader
from torch.autograd import Variable
from torchvision import transforms



class Segmentor():
    def __init__(self) -> None:

        model_dir = 'modelzoo/u2net.pth'

        self.net = U2NET(3, 1)
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_dir))
            self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        self.net.eval()

        self.transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])


    def __call__(self, image: np.ndarray) -> np.ndarray:

        img = torch.from_numpy(image).float()

        sample = self.transform({'imidx': np.array([0]), 'image': img, 'label': img})

        img = sample['image'].unsqueeze(0)

        if torch.cuda.is_available():
            img = Variable(img.cuda())
        else:
            img = Variable(img)

        d1,d2,d3,d4,d5,d6,d7= self.net(img.float())
        
        pred = d1[:,0,:,:]
        pred = self.normPRED(pred)
        predict = pred
        # predict = predict.squeeze()
        predict_np = pred.cpu().data.numpy().squeeze()

        im = Image.fromarray(predict_np*255).convert('RGB')

        im = np.array(im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR))

        return im
    
    def normPRED(self,d: torch.tensor) -> torch.tensor:
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn