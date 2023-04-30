import numpy as np
import cv2
import os
import argparse

from ui.ui import UserInterface
from detector import Detector
from sam_detector import SAMDetector
from reidentification import Reidentification
from util.isar_utils import get_image_it_from_folder
from evaluation import Evaluation

import torch

from params import DATADIR, DATASET, EVALDIR, IMGDIR, TASK, OUTDIR, FPS, EMBDIR, FeatureModes


fps = 10


def main(datadir, outdir, dataset, task, feature_mode_str):
	feature_mode = FeatureModes[feature_mode_str]

	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	use_precomputed_embeddings = True

	if feature_mode == FeatureModes.DETR_CLIP:
		detector = Detector("cpu", "vit_h")
	else: 
		detector = SAMDetector("cpu", "vit_h", use_precomputed_embeddings=True, outdir = outdir, n_per_side=16, feature_mode=feature_mode)
	    
	
	ui = UserInterface(detector=detector)
	if dataset == 'DAVIS' or dataset == 'Habitat_single_obj':
		eval = Evaluation(EVALDIR)


	ious = {}

	images = get_image_it_from_folder(IMGDIR)

	while True:
		image = images.__next__()
		print("\nImage: ", image)
		embedding = os.path.join(EMBDIR, image.replace(".jpg", ".pt"))
		# if os.path.exists(embedding):
		# 	emb = torch.load(embedding)
		img = cv2.imread(os.path.join(IMGDIR, image))
		ui.img = img
		ui.embedding = embedding

		prob, boxes, seg = detector.detect(img, image, embedding)

		if detector.start_reid and (dataset == 'DAVIS' or dataset == 'Habitat_single_obj'):
			iou = eval.compute_IoU(cv2.cvtColor(np.float32(seg), cv2.COLOR_BGR2GRAY) > 0, eval.get_gt_mask(image))
			print("Intersection over Union wrt. ground truth: ", iou)
			ious[image] = iou
			
		show_img = img.copy()

		if detector.start_reid and detector.__class__ == Detector:
			ui.plot_boxes(show_img, prob, boxes)
		
		# if detector.__class__ == SAMDetector and detector.show_images:
		# 	for point in detector.point_grid:
		# 		cv2.circle(show_img, (int(point[1]*show_img.shape[1]), int(point[0]*show_img.shape[0])), radius=4, color=(255, 255, 255), thickness=-1)
		
		cv2.imshow('image', show_img)

		print("Select object to track and press enter.")
		
		if not detector.start_reid:
			k = cv2.waitKey(0)
		else:
			k = cv2.waitKey(int(1000/fps)) #& 0xFF ## TODO: find alternative to waitKey such that operations can be performed in parallel
		if k == 27:
			break
	
	print("Mean IoU: ", np.mean(list(ious.values())))


	cv2.destroyAllWindows()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="A script to process data using datadir and outdir")

	parser.add_argument(
        "-d", "--datadir", type=str, default=DATADIR, help="Path to the data directory (default: 'params/DATADIR')"
    )
	parser.add_argument(
        "-o", "--outdir", type=str, default=OUTDIR, help="Path to the output directory (default: 'params/OUTDIR')"
    )
	parser.add_argument(
        "-ds", "--dataset", type=str, default=DATASET, help="name of the dataset (default: 'params/DATASET'), 'Habitat_single_obj' or 'DAVIS'"
    )
	parser.add_argument(
        "-t", "--task", type=str, default=TASK, help="name of the task (default: 'params/TASK')"
    )
	parser.add_argument(
        "-f", "--feature_mode", type=str, default="CLIP_SAM", choices=["SAM", "CLIP", "CLIP_SAM", "DETR_CLIP"],
    )

	args = parser.parse_args()

	IMGDIR = os.path.join(args.datadir, args.task, "rgb")
	EVALDIR = os.path.join(args.datadir, args.task, "semantics")
	EMBDIR = os.path.join(args.datadir, args.task, "embeddings")

	main(args.datadir, args.outdir, args.dataset, args.task, args.feature_mode)