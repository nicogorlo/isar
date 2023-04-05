import numpy as np
import cv2
import torch
import clip
from PIL import Image

# import sys
# sys.path.append('external/dino-vit-features')
# from extractor import ViTExtractor

class Reidentification():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)        

    def __call__(self, img):
        img_feat = self.extract_img_feat(img)

        return img_feat

    def extract_img_feat(self, img: np.ndarray):
        image = self.preprocess(Image.fromarray(img[:,:,::-1])).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.squeeze()
        
    def get_similarity(self, img_feat1, img_feat2):

        similarity = torch.inner(img_feat1, img_feat2)
        return similarity.item()
    