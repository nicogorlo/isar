# ISAR
Open-Vocabulary **I**nstance **S**egmentation **a**nd **R**e-identification


## Acknowledgements
ISAR builds on and utilizes previous works such as [OW-DETR][ow_detr_link]  (which builds on 
[Deformable DETR][deformable_detr_link], [Detreg][Detreg_link], and [OWOD][OWOD_link]), $U^2$[-Net][u2_net_link], [Segment Anything][SAM_link] and [CLIP][clip_link].

[ow_detr_link]: https://github.com/akshitac8/OW-DETR
[deformable_detr_link]: https://github.com/fundamentalvision/Deformable-DETR
[Detreg_link]: https://github.com/amirbar/DETReg
[OWOD_link]: https://github.com/JosephKJ/OWOD
[u2_net_link]: https://github.com/xuebinqin/U-2-Net
[clip_link]: https://github.com/openai/CLIP
[SAM_link]: https://github.com/facebookresearch/segment-anything


## TODO:
* Improve Bounding Box Proposals
* Extend feature vector
* Support Multi-object tracking and Re-ID
* Clean up code, move reid stuff from detector.py to reidentification.py
