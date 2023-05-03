import numpy as np
import cv2
import os

import torch

from sklearn.decomposition import PCA

class VisualizationPca():
    
    def __init__(self):
        cv2.namedWindow('pca')
        
    def set_template(self, descriptors: np.ndarray, mask: torch.Tensor, patch_w: int, patch_h: int):
        self.mask = mask
        mask_sml = (cv2.resize(mask.squeeze().astype(float), (patch_w, patch_h)) > 0.5)
        mask_sml = mask_sml.reshape(patch_h * patch_w)
        self.template_descriptors = descriptors[mask_sml, :]

        self.patch_w = patch_w
        self.patch_h = patch_h

        self.pca = PCA(n_components=3)
        self.pca.fit(self.template_descriptors)

    def transform(self, descriptors: np.ndarray):
        principal_components = self.pca.transform(descriptors)
        # min max normalization:
        principal_components = self.min_max_normalization(principal_components)

        return principal_components 
    
    def visualize(self, principal_components: np.ndarray, img_shape):
        principal_components = principal_components
        principal_components = principal_components.reshape(self.patch_h, self.patch_w, 3)
        principal_components = cv2.resize(principal_components, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("pca", principal_components.astype("float64"))

        cv2.waitKey(0)

    def min_max_normalization(self,arr):
        # Perform min-max normalization for each channel
        min_vals = np.min(arr, axis=0)
        max_vals = np.max(arr, axis=0)
        normalized_arr = (arr - min_vals) / (max_vals - min_vals)

        return normalized_arr

