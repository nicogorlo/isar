import numpy as np
import cv2
import os
import json
from pathlib import Path
from datetime import datetime
import argparse
from importlib import import_module
import yaml

from evaluation import Evaluation

from tqdm import tqdm

from detector import GenericDetector
from baseline_method import BaselineMethod

from util.isar_utils import performance_measure

class Benchmark():
    def __init__(self, method_config_file: str, outdir: str, datadir: str, device: str="cpu"):
        self.datasets = ['multi_object']
        self.modes = ['single_shot', 'multi_shot']
        self.datadir = datadir

        with open(method_config_file, 'r') as f:
            method_config = yaml.safe_load(f)

        module = method_config["method_class"].rsplit(".", 1)
        Method = getattr(import_module(module[0]), module[1])
        self.detector = Method(device = device, outdir = outdir, **method_config)

        self.dataset = None
        self.stats = {}

        self.detector.show_images = False

        self.outdir = outdir

        if self.detector.show_images:
            cv2.namedWindow("seg")

    "iterate over all datasets (in this version only multi_object)"
    def run(self):
        # self.stats['single_object'] = {}
        self.stats['multi_object'] = {}
        for dataset in self.datasets:
            for mode in self.modes:
                self.run_dataset(dataset, mode)

    def run_dataset(self, dataset: str, mode: str):
        self.dataset = dataset
        if dataset == 'multi_object':
            self.stats[dataset][mode] = self.run_scenario(self.datadir, single_shot=(mode == 'single_shot'))
        else:
            raise Exception("Dataset {} not supported".format(dataset))
    
    def run_scenario(self, datadir: str, single_shot: bool = True) -> dict:
        taskdir = datadir
        dataset_stats = {}
        
        for task in [i for i in sorted(os.listdir(taskdir)) if (".json" not in i)]:
            with performance_measure(f"Task {task}"):
                task_stats = self.run_task(taskdir, task, single_shot=single_shot)
                task_stat_path = os.path.join(self.outdir, task, "task_stats.json")
                with open(task_stat_path, 'w') as f:
                    json.dump(task_stats, f, indent=4)
                dataset_stats.update(task_stats)
            print(f"Task {task} - stats: \n {task_stats}")
        
        return dataset_stats
    
    def run_task(self, taskdir: str, task: str, single_shot: bool = True) -> dict:

        ###################
        # TRAIN SEQUENCES #
        ###################
        with open(os.path.join(taskdir, task, 'info.json'), 'r') as f:
            info = json.load(f)
        semantic_ids = info['semantic_ids']
        color_map = info['color_map']
        train_dir = os.path.join(taskdir, task, 'train/')
        train_scenes = sorted(os.listdir(train_dir))
        prompts = {}

        self.detector.on_new_task(info)
        
        for scene in train_scenes:
            if single_shot:
                with open(os.path.join(train_dir, scene, 'prompts_single.json'), 'r') as f:
                    prompt_dict = json.load(f)
                prompts[scene] = prompt_dict
            else:
                with open(os.path.join(train_dir, scene, 'prompts_multi.json'), 'r') as f:
                    prompt_dict = json.load(f)
                prompts[scene] = prompt_dict
        
        self.detector.train(train_dir, train_scenes, semantic_ids, prompts)
        
        ##################
        # TEST SEQUENCES #
        ##################
        test_dir = os.path.join(taskdir, task, 'test/')
        test_scenes = os.listdir(test_dir)
        task_stats = {}
        
        for scene in [i for i in test_scenes]:
            image_dir = os.path.join(test_dir, scene, "color/")
            eval_dir = os.path.join(test_dir, scene, "semantic_raw/")
            eval = Evaluation(eval_dir, info)
            ious = {}
            self.detector.on_new_test_sequence()
            outdir_scene = os.path.join(self.outdir, task, scene)
            self.detector.outdir = outdir_scene
            if not os.path.exists(self.detector.outdir):
                os.makedirs(self.detector.outdir)

            for image_name in sorted(os.listdir(image_dir)):
                img = cv2.imread(os.path.join(image_dir, image_name))
                emb = os.path.join(test_dir, scene, "embeddings/", image_name.replace(".jpg", ".pt"))
                seg = self.detector.test(img, image_name, emb)

                cv2.waitKey(1)
                eval.compute_evaluation_metrics(seg, image_name)
            scene_stats = eval.report_results(scene)
            task_stats.update(scene_stats)
        return task_stats


def main(method_config: str, outdir: str, datadir: str, device: str) -> None:

    bm = Benchmark(method_config, outdir, datadir, device)
    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H%M%S")
    bm.run()
    stat_path = os.path.join(bm.outdir, f"stats_isar_benchmark_{now_str}.json")
    Path(stat_path).touch(exist_ok=True)
    with open(stat_path, 'w') as f:
        json.dump(bm.stats, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark, computes evaluation metrics for a given dataset")
    
    parser.add_argument("-mc", "--method_config", type=str, default='cfg/baseline_config.yaml', 
                        help='Path to the method config file')
    parser.add_argument(
        "-d", "--datadir", type=str, default="",
        help="Path to the dataset"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default="",
        help="Path to the output directory"
    )
    parser.add_argument(
        "-dev", "--device", type=str, default="cpu", choices=["cpu", "cuda"],
    )

    args = parser.parse_args()
    
    main(args.method_config, args.outdir, args.datadir, args.device)