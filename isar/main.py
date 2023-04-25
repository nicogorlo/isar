import numpy as np
import cv2
import os
from ui.ui import UserInterface
from detector import Detector
from sam_detector import SAMDetector
from reidentification import Reidentification
from util.isar_utils import get_image_it_from_folder
from evaluation import Evaluation

import torch

import sys
sys.path.append('external/dino-vit-features')

from params import DATADIR, DATASET, EVALDIR, IMGDIR, TASK, OUTDIR, FPS, EMBDIR


fps = 10


def main():

	if not os.path.exists(os.path.join("/home/nico/semesterproject/test/", TASK)):
		os.makedirs(os.path.join("/home/nico/semesterproject/test/", TASK))
	
	use_precomputed_embeddings = True
	detector = Detector("cpu", "vit_h", use_precomputed_embeddings)
	detector = SAMDetector("cpu", "vit_h", use_precomputed_embeddings, n_per_side=32)

	ui = UserInterface(detector=detector)
	if DATASET == 'DAVIS' or DATASET == 'Habitat_single_obj':
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

		if detector.start_reid and (DATASET == 'DAVIS' or DATASET == 'Habitat_single_obj'):
			iou = eval.compute_IoU(cv2.cvtColor(np.float32(seg), cv2.COLOR_BGR2GRAY) > 0, eval.get_gt_mask(image))
			print("Intersection over Union wrt. ground truth: ", iou)
			ious[image] = iou
			
		show_img = img.copy()

		if detector.start_reid:
			ui.plot_boxes(show_img, prob, boxes)
		
		if detector.__class__ == SAMDetector and detector.show_images:
			for point in detector.point_grid:
				cv2.circle(show_img, (int(point[1]*show_img.shape[1]), int(point[0]*show_img.shape[0])), radius=4, color=(255, 255, 255), thickness=-1)
		
		cv2.imshow('image', show_img)
		
		if not detector.start_reid:
			k = cv2.waitKey(0)
		else:
			k = cv2.waitKey(int(1000/fps)) #& 0xFF ## TODO: find alternative to waitKey such that operations can be performed in parallel
		if k == 27:
			break
	
	print("Mean IoU: ", np.mean(list(ious.values())))


	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()