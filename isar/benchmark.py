import numpy as np
import cv2
import os
import json
from pathlib import Path

from detector import Detector
from reidentification import Reidentification
from evaluation import Evaluation
from util.isar_utils import get_image_it_from_folder


class Benchmark():
    def __init__(self, outdir):
        self.datasets = ['DAVIS_single_obj', 'Habitat_single_obj']
        self.datadir_DAVIS = "/home/nico/semesterproject/data/DAVIS_single_object_tracking"
        self.datadir_Habitat_single_obj = "/home/nico/semesterproject/data/habitat_single_object_tracking/"

        self.detector = Detector()

        self.stats = {}

        self.detector.show_images = False

        self.detector.segmentor.use_precomputed_embedding = True

        self.outdir = outdir



    def run(self):
        for dataset in self.datasets:
            self.run_dataset(dataset)


    def run_dataset(self, dataset):
        if dataset == 'DAVIS_single_obj':
            self.stats[dataset] = self.run_single_object(self.datadir_DAVIS)
        elif dataset == 'Habitat_single_obj':
            self.stats[dataset] = self.run_single_object(self.datadir_Habitat_single_obj)
        else:
            raise Exception("Dataset {} not supported".format(dataset))
        
    def run_single_object(self, datadir):

        taskdir = datadir

        with open(os.path.join(taskdir, 'prompt_dict.json'), 'r') as f:
            prompt_dict = json.load(f)

        dataset_stats = {}

        
        for task in [i for i in sorted(os.listdir(taskdir)) if ".json" not in i][:50]:
            print("Task: ", task)

            imgdir = os.path.join(taskdir, task, 'rgb/')
            evaldir = os.path.join(taskdir, task, 'semantics/')
            embdir = os.path.join(taskdir, task, 'embeddings/')
            task_stats = self.run_task(task, imgdir, evaldir, embdir, prompt_dict[task])
            dataset_stats[task] = task_stats

        return dataset_stats
        
    def run_task(self, task, imgdir, evaldir, embdir, prompt=None):

        self.detector.outdir = os.path.join(self.outdir, task)
        
        if not os.path.exists(os.path.join(self.outdir, task)):
            os.makedirs(os.path.join(self.outdir, task))
        eval = Evaluation(evaldir)
        ious = {}

        image_names = sorted(os.listdir(imgdir))

        if prompt is not None:
            img0 = cv2.imread(os.path.join(imgdir, image_names[0]))
            emb0 = os.path.join(embdir, image_names[0].replace(".jpg", ".pt"))
            prob, boxes, seg = self.detector.detect(img0, image_names[0], embedding=emb0)
            cutout, seg, freeze, selected_prob, selected_box = self.detector.on_click(x = prompt['x'], y = prompt['y'], img = img0, embedding=emb0)
            image_names.pop(0)


        for image_name in image_names:
            img = cv2.imread(os.path.join(imgdir, image_name))
            emb = os.path.join(embdir, image_name.replace(".jpg", ".pt"))

            prob, boxes, seg = self.detector.detect(img, image_name, emb)

            if self.detector.start_reid:
                eval.compute_evaluation_metrics(cv2.cvtColor(np.float32(seg), cv2.COLOR_BGR2GRAY) > 0, eval.get_gt_mask(image_name), image_name)

        task_stats = eval.report_results(task)
    
        return task_stats

def main():

    bm = Benchmark("/home/nico/semesterproject/test/")
    bm.run()
    
    stat_path = os.path.join(bm.outdir, "stats.json")
    Path(stat_path).touch(exist_ok=True)
    with open(stat_path, 'w') as f:
        json.dump(bm.stats, f, indent=4)

if __name__ == "__main__":
    main()