import numpy as np
import cv2
import os

import torch
from torch import nn
import torchvision.transforms as T
torch.set_grad_enabled(False)

class GenericDetector():
    """
    Generic Detector class
    functions are called in benchmark.py, are meant to be overwritten in child classes
    """
    def __init__(self, *args, **kwargs):
        self.device = kwargs.get("device", "cpu")

    def train_single_shot(self, train_data, annotations):
        """
        train a single shot detector
        train data: single image
        annotations: one bounding box and point prompt per object class 
            structure:
                {
                    image_name: {f"{instance_class}": 
                                    {"point_prompt": [x, y], "bbox": [x1, y1, x2, y2]}
                                , ...}
                    , ...
                    })
        """
        print("training for single shot detection")
        
        return
    
    def train_multi_shot(self, train_data, annotations):
        """
        train a single shot detector
        train data: sequence of images
        annotations: a few bounding boxes and point prompts per object class 
        """
        print("training for multi shot detection")
        return
    
    def on_new_task(self, *args, **kwargs):
        """
        reset detector for new task (new train sequences)
        """
        print("new task")
        return

    
    def on_new_test_sequence(self, *args, **kwargs):
        """
        reset detector for new test sequence
        """
        print("new test sequence")
        return
    
    def inference(self, img: np.ndarray, image_name: str):
        """
        called for every image in a test sequence

        args: image as numpy array
        returns: segmentation prediction as numpy array
        """
        return np.zeros(img.shape)