import json
import os
import numpy as np
from pathlib import Path
import random
import colorsys
import argparse


def generate_color_palette(num_colors):
    random.seed(42) 
    hsv_tuples = [(x / num_colors, 1., 1.) for x in range(num_colors)]
    random.shuffle(hsv_tuples)
    rgb_tuples = map(lambda x: tuple(int(255 * i) for i in colorsys.hsv_to_rgb(*x)), hsv_tuples)
    bgr_tuples = map(lambda x: (x[2], x[1], x[0]), rgb_tuples)
    return list(bgr_tuples)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Store Task Info")
    parser.add_argument("--datadir", type=str, default="")
    args = parser.parse_args()

    datadir = args.datadir
    info = {}
    prompts = {}

    for task in [i for i in sorted(os.listdir(datadir))]:
        color_map = {}
        prompts = {}
        for file in Path(os.path.join(datadir, task)).rglob('prompts_multi.json'):
            with file.open() as f:
                prompts.update(json.load(f))

        for file in Path(os.path.join(datadir, task)).rglob('color_map.json'):
            with file.open() as f:
                color_map.update(json.load(f))
            
        if len(color_map) == 0:
            labels = sorted([int(i) for i in np.unique([int(item) for sublist in [list(p.keys()) for p in prompts.values()] for item in sublist])])
            semantic_palette = generate_color_palette(np.unique(labels).shape[0])
            for i, label in enumerate(labels):
                color_map.update({str(label): semantic_palette[i%len(semantic_palette)]})

        info = {"task_name": task, 
                "semantic_ids": [int(i) for i in np.unique([int(item) for sublist in [list(p.keys()) for p in prompts.values()] for item in sublist])],
                "num_prompts_multi": len(prompts),
                "color_map": color_map}
        print(info)

        with open(os.path.join(datadir, task, "info.json"), "w") as f:
            json.dump(info, f)

        for scene in sorted(os.listdir(os.path.join(datadir, task, "test"))):
            with open(os.path.join(datadir, task, "test", scene, "color_map.json"), "w") as f:
                json.dump(color_map, f)