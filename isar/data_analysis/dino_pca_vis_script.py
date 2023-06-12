import numpy as np

import torch
import torch.nn as nn
import timm
from PIL import Image
import cv2
import torchvision.transforms as T

from sklearn.decomposition import PCA

patch_h = 40
patch_w = 40

def load_dino_v2_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model
    

def fit_pca(features, mask):
    mask_sml = (cv2.resize(mask.squeeze().astype(float), (patch_w, patch_h)) > 0.5)
    mask_sml = mask_sml.reshape(patch_h * patch_w)
    features_mask = features[mask_sml, :]

    pca = PCA(n_components=3)
    pca.fit(features_mask.cpu().numpy())
    
    return pca

def preprocess_image_dino(image: np.ndarray):
    transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).cuda()
    return tensor

def extract_features_dino(model, input_tensor):
    with torch.no_grad():
        features_dict = model.forward_features(input_tensor)
        features = features_dict['x_norm_patchtokens']
    return features.squeeze()

def img_to_pca_features_dino(model, pca, image):
    input_tensor = preprocess_image_dino(image)
    features = extract_features_dino(model, input_tensor)

    pca_features = pca.transform(features.cpu().numpy())
    pca_features = pca_features.reshape(patch_h, patch_w, 3)
    pca_features = cv2.resize(pca_features, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # min max normalization for each channel:
    pca_features = cv2.normalize(pca_features, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return pca_features

# draws a grid over the image for the corresponding patches
def overlay_grid(image, grid_size):

    height, width, _ = image.shape
    
    for i in range(1, grid_size[0]):
        cv2.line(image, (width * i // grid_size[0], 0), (width * i // grid_size[0], height), (255, 255, 255), 1)

    for i in range(1, grid_size[1]):
        cv2.line(image, (0, height * i // grid_size[1]), (width, height * i // grid_size[1]), (255, 255, 255), 1)

from evaluation import Evaluation
import os

def main():
    DATADIR = ""
    eval = Evaluation(os.path.join(DATADIR, "semantics"))
    image_names = sorted(os.listdir(os.path.join(DATADIR, "rgb")))
    gt_mask = eval.get_gt_mask(image_names[0])

    model = load_dino_v2_model()
    image = cv2.imread(os.path.join(DATADIR, "rgb", image_names[0]))
    input_tensor = preprocess_image_dino(image)
    features = extract_features_dino(model, input_tensor)

    pca = fit_pca(features, gt_mask)

    pca_features = pca.transform(features.cpu().numpy())
    pca_features = pca_features.reshape(patch_h, patch_w, 3)
    pca_features = cv2.resize(pca_features, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # min max normalization for each channel:
    pca_features = cv2.normalize(pca_features, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.namedWindow("pca")
    cv2.namedWindow("image")
    cv2.imshow("pca", pca_features)

    cv2.imshow("image", image)

    cv2.waitKey(0)

    image_names.pop(0)
    for image_name in image_names:
        image = cv2.imread(os.path.join(DATADIR, "rgb", image_name))
        pca_features = img_to_pca_features_dino(model, pca, image)

        cv2.namedWindow("pca")
        cv2.namedWindow("image")
        cv2.imshow("pca", pca_features)
        cv2.imshow("image", image)

        k = cv2.waitKey(int(1000 / 30))
        if k == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
