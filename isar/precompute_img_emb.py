import numpy as np
import cv2
import os
from tqdm import tqdm

import torch
torch.set_grad_enabled(False)

# from util.isar_utils import performance_measure

from segment_anything import sam_model_registry, SamPredictor



# very basic. So far sequential, batching would be way better


class PrecomputeImgEmbedding():
    def __init__(self):
        self.datasets = ['DAVIS_single_obj', 'Habitat_single_obj']
        self.datadir_DAVIS = "/home/gorlon/semesterproject/data/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/"
        self.datadir_Habitat_single_obj = "/home/gorlon/semesterproject/data/habitat_single_object_tracking/"

        sam_checkpoint = "~/home/gorlon/semesterproject/modelzoo/sam_vit_h_4b8939.pth"
        device = "cuda"
        model_type = "vit_h"

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

                # with performance_measure("set image - task:{}; image name:{}".format(task, image_name)):                
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
                
                # with performance_measure("set image - task:{}; image name:{}".format(task, image_name)): 
                self.predictor.set_image(image, image_format='BGR')

                features = self.predictor.get_image_embedding()

                torch.save(features, out_path)


def main():
    precompute_img_emb = PrecomputeImgEmbedding()
    precompute_img_emb.precompute_img_emb()


if __name__ == "__main__":
    main()