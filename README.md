# ISAR
Open-Vocabulary **I**nstance **S**egmentation **a**nd **R**e-identification Benchmark


## Acknowledgements
The ISAR benchmark dataset is a synthetic dataset built with the [AI-Habitat Simulator][habitat_link] using data of the [Replica-Dataset][Replica_link], the [habitat-matterport-3D-dataset][hm3d_link] and the [ycb-object-and-model-set][ycb_link]. Further it uses the objects mentioned in ./isar/attribution/README.md .

The baseline method of ISAR builds on and utilizes previous works such as [Segment Anything][SAM_link] and [DINOv2][dino_link]. 
Legacy versions of the method work on [OW-DETR][ow_detr_link]  (which builds on 
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

When using the dataset in your research, please also cite: [//]: #add citations
* [AI-Habitat Simulator][habitat_link]
* [Replica-Dataset][Replica_link]
* [habitat-matterport-3D-dataset][hm3d_link]
* [ycb-object-and-model-set][ycb_link]


## Folder structure:
the datasets are structured as follows:

Dataset_name <br>
|--single_object/multi_object <br>
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
--------|--semantics (this is used for visualization) <br>
-----------|--xxxxxxx.png <br>
-----------|-- ... <br>
--------|--semantics_raw (this is used for eval) <br>
-----------|--xxxxxxx.npy <br>
-----------|-- ... <br>
------|-- ... <br>
--|-- ... <br>