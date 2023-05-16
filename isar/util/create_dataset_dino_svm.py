import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image
import random
import pickle

import torch
import torchvision.transforms as T
from torch.utils.data import RandomSampler
torch.set_grad_enabled(False)

from segment_anything import sam_model_registry, SamPredictor

"""
Precomputes the SAM image embeddings for the DAVIS and Habitat datasets.
"""
class CreateDinoDataset():
    def __init__(self, datadir_davis, datadir_habitat, dino_type):
        self.datadirs = {'DAVIS_single_obj': datadir_davis, 'Habitat_single_obj': datadir_habitat}

        self.features_per_image = 20
        self.images_per_task = 20
        self.img_size = 512
        self.patch_h = 40
        self.patch_w = 40

        self.dino = self.load_dino_v2_model(dino_type)

        self.feature_dict = {}

    def create_dataset_all_data(self):
        for dataset in self.datadirs.keys():
            dataset_feature_dict = self.compute_sampled_features(dataset)

            self.feature_dict[dataset] = dataset_feature_dict

    def compute_sampled_features(self, dataset):
        taskdir = self.datadirs[dataset]

        dataset_feature_dict = {}
        for task in tqdm([i for i in sorted(os.listdir(taskdir)) if ".json" not in i]):
            dataset_feature_dict[task] = {}

            imgdir = os.path.join(taskdir, task, 'rgb/')

            for image_name in random.sample(os.listdir(imgdir), k=self.images_per_task):
                img_path = os.path.join(imgdir, image_name)

                img = cv2.imread(img_path)

                tensor_in = self.preprocess_image_dino(img)
                features = self.extract_features_dino(self.dino, tensor_in)

                indices = torch.randperm(features.shape[0])[:self.features_per_image]

                # show location of indices:
                for i in indices:
                    coords = (i % self.patch_w * self.img_size/self.patch_w, i // self.patch_h * self.img_size/self.patch_h)
                    coords = (int(coords[0]* img.shape[1]/self.img_size), int(coords[1]* img.shape[0]/self.img_size))
                    cv2.circle(img, coords, 5, (0, 0, 255), -1)
                # cv2.imshow("img", img)
                # cv2.waitKey(15)

                features_sampled = features[indices, :]

                dataset_feature_dict[task][image_name] = features_sampled
     

        return dataset_feature_dict

    def load_dino_v2_model(self, dino_model_type):
        model = torch.hub.load('facebookresearch/dinov2', dino_model_type)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model
    
    def preprocess_image_dino(self, image: np.ndarray):
        transform = T.Compose([
        T.Resize((self.patch_h * 14, self.patch_w * 14)),
        T.CenterCrop((self.patch_h * 14, self.patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = transform(image).unsqueeze(0).cuda()
        return tensor
    
    def extract_features_dino(self, model, input_tensor):
        with torch.no_grad():
            features_dict = model.forward_features(input_tensor)
            features = features_dict['x_norm_patchtokens']
            # features = features.reshape(patch_h, patch_w, feat_dim)
        return features.squeeze()


import argparse

def main():
    parser = argparse.ArgumentParser(description="Precompute image embeddings")

    parser.add_argument(
        "-dd", "--datadir_davis", type=str, default="/home/nico/semesterproject/data/DAVIS_single_object_tracking/", 
        help="Path to the DAVIS dataset"
    )
    parser.add_argument(
        "-dh", "--datadir_habitat", type=str, default="/home/nico/semesterproject/data/habitat_single_object_tracking/", 
        help="Path to the Habitat dataset"
    )
    parser.add_argument(
        "-di", "--dino_type", type=str, default="dinov2_vitl14",
        help="type of the dino model to use"
    )

    args = parser.parse_args()

    create_dino_dataset = CreateDinoDataset(args.datadir_davis, args.datadir_habitat, args.dino_type)
    create_dino_dataset.create_dataset_all_data()

    with open(f"../../../data/dino_features_{args.dino_type}.pkl", "wb") as f:
        pickle.dump(create_dino_dataset.feature_dict, f)

    with open(f"../feature_dict_dinov2/dino_features_{args.dino_type}.pkl", "wb") as f:
        pickle.dump(create_dino_dataset.feature_dict, f)


if __name__ == "__main__":
    main()