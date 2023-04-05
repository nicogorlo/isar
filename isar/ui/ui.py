import numpy as np
import cv2
import os
import torch

from detector import Detector

datadir = "/home/nico/semesterproject/data/videos/3dod/Training/47670305/47670305_frames/lowres_wide/"

class UserInterface():
    def __init__(self, detector) -> None:
        # self.dataset = "/home/nico/semesterproject/data/videos/3dod/Training/47670305/47670305_frames/lowres_wide"
        self.ix,self.iy = -1,-1
        self.datadir = datadir

        self.fps = 15 #1/s
        self.reid_on = False

        cv2.namedWindow('image')
        cv2.namedWindow('freeze')
        cv2.namedWindow('cutout')
        cv2.namedWindow('seg')
        cv2.namedWindow('match')
        cv2.setMouseCallback('image',self.select)

        cv2.moveWindow('image', 0, 0)
        cv2.moveWindow('freeze', 450, 0)
        cv2.moveWindow('cutout', 850, 0)
        cv2.moveWindow('seg', 850, 220)
        cv2.moveWindow('match', 850, 500)

        self.img = np.zeros((192,256,3), np.uint8)
        self.detect_frame = np.zeros((192,256,3), np.uint8)
        cv2.imshow('freeze', self.detect_frame)
        cv2.imshow('image', self.img)
        
        self.detector = detector


    def select(self, event, x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.reid_on = True
            self.ix,self.iy = x,y
            cutout, seg, freeze, selected_prob, selected_box = self.detector.on_click(x, y, self.img)

            cv2.imshow('cutout', cutout)
            cv2.imshow('seg', seg)
            self.plot_single_box(self.img, selected_prob, selected_box)
            cv2.imshow('freeze', freeze)


    def plot_single_box(self, img, prob, box):
        xmin, ymin, xmax, ymax = box[0]
        xmin=int(xmin); xmax=int(xmax); ymin=int(ymin); ymax=int(ymax)
        cv2.rectangle(img, pt1=np.array([xmin, ymin]), pt2=np.array([xmax,ymax]),
                                color=(0,255,0), thickness=2)


    def plot_boxes(self, img, prob, boxes):
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), self.detector.COLORS * 100):
            xmin=int(xmin); xmax=int(xmax); ymin=int(ymin); ymax=int(ymax)
            cv2.rectangle(img, pt1=np.array([xmin, ymin]), pt2=np.array([xmax,ymax]),
                                    color=c, thickness=1)
            # cl = p.argmax()
            # text = f'{self.detector.CLASSES[cl]}: {p[cl]:0.2f}'
            # cv2.putText(img = img, org = (xmin, ymin-2), text=text, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.9, color=c, thickness=2)


    def draw(self, mask):
        color = np.array([0,255,0], dtype='uint8')
        masked_img = np.where(mask[...,None], color, self.img)
        self.detect_frame = cv2.addWeighted(self.detect_frame, 0.8, masked_img, 0.2,0)
        self.img = cv2.addWeighted(self.img, 0.8, masked_img, 0.2,0)