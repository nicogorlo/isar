import numpy as np
import cv2
import os
import json
from pathlib import Path
from datetime import datetime
import argparse

from detector import Detector
from sam_detector import SAMDetector
from dino_detector import DinoDetector
from reidentification import Reidentification
from evaluation import Evaluation
from util.isar_utils import get_image_it_from_folder

from params import FeatureModes


class Benchmark():
    def __init__(self, outdir, datadir_davis, datadir_habitat, feature_mode=FeatureModes.CLIP_SAM):
        self.datasets = ['DAVIS_single_obj', 'Habitat_single_obj']
        self.datadir_DAVIS = datadir_davis
        self.datadir_Habitat_single_obj = datadir_habitat

        if feature_mode == FeatureModes.DETR_CLIP:
            self.detector = Detector("cpu", "vit_h")
        elif feature_mode == FeatureModes.DINO_SVM:
            self.detector = DinoDetector("cpu", "vit_h", "dinov2_vitl14", True, outdir, n_per_side=16)
        else: 
            self.detector = SAMDetector("cpu", "vit_h", use_precomputed_embeddings=True, outdir = outdir, n_per_side=16, feature_mode=feature_mode)
        
        self.dataset = None
        self.stats = {}

        self.detector.show_images = True

        self.outdir = outdir

        if self.detector.show_images:
            cv2.namedWindow("seg")

        self.use_gt_mask_first_image = False
        self.print_gt_feature_distance = False

    "iterate over all datasets (DAVIS_single_obj, Habitat_single_obj)"
    def run(self):
        for dataset in self.datasets:
            self.run_dataset(dataset)

    def run_dataset(self, dataset):
        self.dataset = dataset
        if dataset == 'DAVIS_single_obj':
            self.stats[dataset] = self.run_single_object(self.datadir_DAVIS)
        elif dataset == 'Habitat_single_obj':
            self.stats[dataset] = self.run_single_object(self.datadir_Habitat_single_obj)
        else:
            raise Exception("Dataset {} not supported".format(dataset))
        
    "iterate over all tasks in a dataset (e.g. 'car', 'duck', 'giraffe', ...)"        
    def run_single_object(self, datadir):

        taskdir = datadir

        with open(os.path.join(taskdir, 'prompt_dict.json'), 'r') as f:
            prompt_dict = json.load(f)

        dataset_stats = {}

        for task in [i for i in sorted(os.listdir(taskdir)) if (".json" not in i and "blackswan" in i)]:
            print("Task: ", task)

            imgdir = os.path.join(taskdir, task, 'rgb/')
            evaldir = os.path.join(taskdir, task, 'semantics/')
            embdir = os.path.join(taskdir, task, 'embeddings/')
            task_stats = self.run_task(task, imgdir, evaldir, embdir, prompt_dict[task])
            dataset_stats[task] = task_stats

        return dataset_stats
    
    "iterate over all images in a task (e.g. '0000000.jpg', '0000001.jpg', ...)"
    def run_task(self, task, imgdir, evaldir, embdir, prompt=None):

        self.detector.outdir = os.path.join(self.outdir, task)
        
        if not os.path.exists(os.path.join(self.outdir, task)):
            os.makedirs(os.path.join(self.outdir, task))
        eval = Evaluation(evaldir)
        ious = {}

        image_names = sorted(os.listdir(imgdir))

        """
        set prompt according to prompt_dict.json
        """
        if prompt is not None:
            img0 = cv2.imread(os.path.join(imgdir, image_names[0]))
            emb0 = os.path.join(embdir, image_names[0].replace(".jpg", ".pt"))
            if self.detector.__class__ == DinoDetector and self.use_gt_mask_first_image:
                mask = eval.get_gt_mask(image_names[0])
                cutout, seg, freeze, selected_prob, selected_box = self.detector.on_click(
                    x = prompt['x'], y = prompt['y'], img = img0, embedding_sam=emb0, dataset = self.dataset, task = task,
                    gt_mask= mask
                    )     
            else:
                prob, boxes, seg = self.detector.detect(img0, image_names[0], embedding=emb0)
                if self.detector.__class__ == DinoDetector: 
                    cutout, seg, freeze, selected_prob, selected_box = self.detector.on_click(
                        x = prompt['x'], y = prompt['y'], img = img0, embedding_sam=emb0, dataset = self.dataset, task = task
                        )
                else: 
                    cutout, seg, freeze, selected_prob, selected_box = self.detector.on_click(
                        x = prompt['x'], y = prompt['y'], img = img0, embedding=emb0
                        )


            """
            for debug: compute template feature of gt mask of first image
            * assumes, that gt mask of first image can be obtained
            * overwrites variables set by on_click
            """
            if self.detector.__class__ == SAMDetector and self.use_gt_mask_first_image:
                mask = eval.get_gt_mask(image_names[0])
                img_features = self.detector.predictor.get_image_embedding().squeeze().cpu().numpy()
                self.detector.template_feature = self.detector.mask_to_features(img0, mask, img_features)
            image_names.pop(0)

        """
        iterate over all images in a task (e.g. '0000000.jpg', '0000001.jpg', ...)
        """
        for image_name in image_names:
            img = cv2.imread(os.path.join(imgdir, image_name))
            emb = os.path.join(embdir, image_name.replace(".jpg", ".pt"))

            prob, boxes, seg = self.detector.detect(img, image_name, emb)

            if self.detector.start_reid:
                eval.compute_evaluation_metrics(cv2.cvtColor(np.float32(seg), cv2.COLOR_BGR2GRAY) > 0, eval.get_gt_mask(image_name), image_name)

            """
            debug: output distance of feature of gt mask to template feature
            """
            if self.detector.__class__ == SAMDetector and self.print_gt_feature_distance:

                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
                self.detector.set_img_embedding(img, emb)
                mask = eval.get_gt_mask(image_name)
                mask = cv2.resize(mask.astype("uint8"), (512, 512), interpolation=cv2.INTER_NEAREST)
                if np.sum(mask) > 0: 
                    img_features = self.detector.predictor.get_image_embedding().squeeze().cpu().numpy()
                    gt_mask_descriptor = self.detector.mask_to_features(img, mask, img_features)
                    nearest_neighbor_index, nearest_neighbor_descriptor, min_distance = self.detector.get_nearest_neighbor_descriptor([gt_mask_descriptor])
                    print("distance of gt to template: ", min_distance)
                self.detector.predictor.reset_image()

            if self.detector.show_images:
                cv2.waitKey(1)

        task_stats = eval.report_results(task)
    
        return task_stats


def main(outdir, datadir_davis, datadir_habitat, feature_mode_str):
    feature_mode = FeatureModes[feature_mode_str]
    
    bm = Benchmark(outdir, datadir_davis, datadir_habitat, feature_mode)
    bm.use_gt_mask_first_image = True
    bm.print_gt_feature_distance = True
    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H%M%S")
    bm.run_dataset('DAVIS_single_obj')
    stat_path = os.path.join(bm.outdir, f"stats_{now_str}_DAVIS_single_obj_pigs.json")
    Path(stat_path).touch(exist_ok=True)
    with open(stat_path, 'w') as f:
        json.dump(bm.stats, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark, computes evaluation metrics for a DAVIS and a Habitat dataset")

    parser.add_argument(
        "-dd", "--datadir_davis", type=str, default="/home/nico/semesterproject/data/DAVIS_single_object_tracking/", 
        help="Path to the DAVIS dataset"
    )
    parser.add_argument(
        "-dh", "--datadir_habitat", type=str, default="/home/nico/semesterproject/data/habitat_single_object_tracking/", 
        help="Path to the Habitat dataset"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default="/home/nico/semesterproject/test/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "-f", "--feature_mode", type=str, default="DINO_SVM", choices=["SAM", "CLIP", "CLIP_SAM", "DETR_CLIP", "DINO_SVM"],
    )

    args = parser.parse_args()
    
    main(args.outdir, args.datadir_davis, args.datadir_habitat, args.feature_mode)