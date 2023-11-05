import numpy as np
import cv2
import os
from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch import nn
import torchvision.transforms as T
torch.set_grad_enabled(False)

class GenericDetector(ABC):
    """
    Generic Detector class
    functions are called in benchmark.py, are meant to be overwritten in child classes
    """
    def __init__(self, *args, **kwargs):
        self.device = kwargs.get("device", "cpu")

    @abstractmethod
    def train(self, train_dir: str, train_scenes: list, semantic_ids: list, prompts: dict) -> None:
        """
        train a method
        train_dir: path to train data
        train_scenes: list of scene names
        semantic_ids: list of semantic ids to track
        prompts: dict of prompts for each scene
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
    
    def on_new_task(self, info: dict, *args, **kwargs) -> None:
        """
        reset detector for new task (new train sequences)
        info is the info.json file for the task
        """
        print("new task")
        return

    
    def on_new_test_sequence(self, *args, **kwargs) -> None:
        """
        reset detector for new test sequence
        """
        print("new test sequence")
        return
    
    @abstractmethod
    def test(self, img: np.ndarray, image_name: str, embedding: str = '') -> np.ndarray:
        """
        called sequentially for every image in a test sequence

        args: image as numpy array
        returns: segmentation prediction as numpy array
        """
        return np.zeros(img.shape)