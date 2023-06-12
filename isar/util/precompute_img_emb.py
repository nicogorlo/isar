import numpy as np
import cv2
import os
from tqdm import tqdm

import torch
torch.set_grad_enabled(False)


from segment_anything import sam_model_registry, SamPredictor

"""
Precomputes the SAM image embeddings for the DAVIS and Habitat datasets.
"""
class PrecomputeImgEmbedding():
    def __init__(self, datadir_davis, datadir_habitat, sam_checkpoint):
        self.datasets = ['DAVIS_single_obj', 'Habitat_single_obj']
        self.datadir_DAVIS = datadir_davis
        self.datadir_Habitat_single_obj = datadir_habitat


        device = "cuda"
        model_type = "vit_h"

        self.img_size = 512

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def precompute_img_emb(self):
        for dataset in self.datasets:
            if dataset == 'DAVIS':
                self.datadir = self.datadir_DAVIS
                self.precompute_img_emb_DAVIS()
            elif dataset == 'Habitat_single_obj':
                self.datadir = self.datadir_Habitat_single_obj
                self.precompute_img_emb_Habitat_single_obj()
            else:
                raise Exception("Unknown dataset")

    def precompute_img_emb_DAVIS(self):
        taskdir = os.path.join(self.datadir, "JPEGImages/480p/")
        outdir = os.path.join(self.datadir, "ImageEmbeddings/")

        for task in tqdm([i for i in sorted(os.listdir(taskdir)) if ".json" not in i]):
            print("Task: ", task)

            imgdir = os.path.join(taskdir, task)
            outdir_task = os.path.join(outdir, task)

            if not os.path.exists(outdir_task):
                os.makedirs(outdir_task)

            for image_name in sorted(os.listdir(imgdir)):
                img_path = os.path.join(imgdir, image_name)

                out_path = os.path.join(outdir_task, image_name.replace(".jpg", ".pt"))

                if os.path.exists(out_path):
                    continue

                image = cv2.imread(img_path)

                image = cv2.resize(image, (self.img_size, self.img_size))
               
                self.predictor.set_image(image, image_format='BGR')

                features = self.predictor.get_image_embedding()

                torch.save(features, out_path)

    

    def precompute_img_emb_Habitat_single_obj(self):
        taskdir = self.datadir_Habitat_single_obj

        for task in tqdm([i for i in sorted(os.listdir(taskdir)) if ".json" not in i]):

            imgdir = os.path.join(taskdir, task, 'rgb/')

            outdir_task = os.path.join(taskdir, task, 'ImageEmbeddings/')

            if not os.path.exists(outdir_task):
                os.makedirs(outdir_task)

            for image_name in sorted(os.listdir(imgdir)):
                img_path = os.path.join(imgdir, image_name)

                out_path = os.path.join(outdir_task, image_name.replace(".jpg", ".pt"))

                if os.path.exists(out_path):
                    continue

                image = cv2.imread(img_path)

                image = cv2.resize(image, (self.img_size, self.img_size))
                
                self.predictor.set_image(image, image_format='BGR')

                features = self.predictor.get_image_embedding()

                torch.save(features, out_path)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Precompute image embeddings")

    parser.add_argument(
        "-dd", "--datadir_davis", type=str, default="", 
        help="Path to the DAVIS dataset"
    )
    parser.add_argument(
        "-dh", "--datadir_habitat", type=str, default="", 
        help="Path to the Habitat dataset"
    )
    parser.add_argument(
        "-cp", "--checkpoint", type=str, default="../modelzoo/sam_vit_h_4b8939.pth",
        help="Path to the output directory"
    )

    args = parser.parse_args()

    precompute_img_emb = PrecomputeImgEmbedding(args.datadir_davis, args.datadir_habitat, args.checkpoint)
    precompute_img_emb.precompute_img_emb()


if __name__ == "__main__":
    main()