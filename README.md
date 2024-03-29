# ISAR
Open-Vocabulary **I**nstance **S**egmentation **a**nd **R**e-identification Benchmark

For the development of effective 3D perception and object-level SLAM systems, it's crucial to reliably detect, segment, and re-identify objects. Moreover, achieving this shouldn't necessitate the use of millions of training examples. We've created this benchmark to highlight this issue and expedite research into algorithms for single-shot and few-shot object instance segmentation and re-identification.

## Getting started
This work has been tested in a python 3.9 environment.

1. Install dependencies

    ```
    pip install cython
    pip install -r requirements.txt
    ```

2. Download Dataset (>70GB)

dataset release pending

## Replicate results

1. Download model weights

    ```
    python3 ./isar/util/download_model_weights.py
    ```

2. run benchmark

    ```
    python3 benchmark.py (...)
    ```

## Benchmark

CLI arguments:

* mc, method_config - path to method config file
* d, datadir - path to directory of dataset
* o, outdir - path to output directory
* -dev, device - device to use. Choices: ["cpu", "cuda"]

## Implement new method:

The easiest way to test a new method on the dataset and recieve results in the same format as the baseline method is:
1. Create new class inheriting from detector.GenericDetector
2. Implement all member functions
3. Replace the detector in benchmark.Benchmark with your own implementation


## Folder structure:
the datasets are structured as follows:

Dataset_name <br>
|--multi_object <br>
--|--task_name <br>
----|--info.json (task info) <br>
----|--train <br>
------|--scene_name<br>
--------|--attributes.json (scene attributes) <br>
--------|--camera_poses.json (6DOF camera pose of each frame) <br>
--------|--color_map.json (unique mapping: semantic_id->rgb_color of scene) <br>
--------|--prompts_single.json (prompts for single-shot case) <br>
--------|--prompts_multi.json (prompts for multi-shot case) <br>
--------|--rgb <br>
-----------|--xxxxxxx.jpg <br>
-----------|-- ... <br>
--------|--(optional:depth) <br>
-----------|--xxxxxxx.png <br>
-----------|-- ... <br>
------|-- ... <br>
----|--test <br>
------|--scene_name<br>
--------|--attributes.json (scene attributes) <br>
--------|--camera_poses.json (6DOF camera pose of each frame) <br>
--------|--color_map.json (unique mapping: semantic_id->rgb_color of scene) <br>
--------|--rgb <br>
-----------|--xxxxxxx.jpg <br>
-----------|-- ... <br>
--------|--(optional:depth) <br>
-----------|--xxxxxxx.png <br>
-----------|-- ... <br>
--------|--semantic (this is used for visualization) <br>
-----------|--xxxxxxx.png <br>
-----------|-- ... <br>
--------|--semantic_raw (this is used for eval) <br>
-----------|--xxxxxxx.png <br>
-----------|-- ... <br>
------|-- ... <br>
--|-- ... <br>


## Acknowledgements
The ISAR benchmark dataset is a synthetic dataset built with the [AI-Habitat Simulator][habitat_link] using data of the [Replica-Dataset][Replica_link], the [habitat-matterport-3D-dataset][hm3d_link] and the [ycb-object-and-model-set][ycb_link]. Further it uses the objects mentioned in ./isar/attribution/README.md .

The baseline method of ISAR builds on and utilizes previous works such as [Segment Anything][SAM_link] and [DINOv2][dino_link]. 
Legacy versions of the method build on [OW-DETR][ow_detr_link]  (which builds on 
[Deformable DETR][deformable_detr_link], [Detreg][Detreg_link], and [OWOD][OWOD_link]) and [CLIP][clip_link].

[ow_detr_link]: https://github.com/akshitac8/OW-DETR
[deformable_detr_link]: https://github.com/fundamentalvision/Deformable-DETR
[Detreg_link]: https://github.com/amirbar/DETReg
[OWOD_link]: https://github.com/JosephKJ/OWOD
[clip_link]: https://github.com/openai/CLIP
[SAM_link]: https://github.com/facebookresearch/segment-anything
[dino_link]: https://github.com/facebookresearch/dinov2
[habitat_link]: https://github.com/facebookresearch/habitat-sim
[Replica_link]: https://github.com/facebookresearch/Replica-Dataset
[hm3d_link]: https://github.com/facebookresearch/habitat-matterport3d-dataset
[ycb_link]: https://www.ycbbenchmarks.com/

When using the dataset in your research, please also cite:
* [AI-Habitat Simulator][habitat_link]
* [Replica-Dataset][Replica_link]
* [habitat-matterport-3D-dataset][hm3d_link]
* [ycb-object-and-model-set][ycb_link]
