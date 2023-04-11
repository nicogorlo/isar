import numpy as np
import cv2
import os
from ui.ui import UserInterface
from detector import Detector
from reidentification import Reidentification
from util.isar_utils import get_image_it_from_folder
from evaluation import Evaluation

import sys
sys.path.append('external/dino-vit-features')

from params import DATADIR
from params import DATASET
from params import TASK
from params import EVALDIR
from params import IMGDIR
from params import OUTDIR


fps = 15


def main():

	if not os.path.exists(os.path.join("/home/nico/semesterproject/test/", TASK)):
		os.makedirs(os.path.join("/home/nico/semesterproject/test/", TASK))
	

	detector = Detector()
	ui = UserInterface(detector=detector)
	if DATASET == 'DAVIS':
		eval = Evaluation(DATADIR, EVALDIR, DATASET)

	ious = {}

	# ui.play_from_folder()
	images = get_image_it_from_folder(os.path.join(DATADIR, IMGDIR), fps)

	while True:
		image = images.__next__()
		img = cv2.imread(os.path.join(DATADIR, IMGDIR, image))
		# img = cv2.resize(img, (285,160))
		ui.img = img
		prob, boxes, seg = detector.detect(img, image)

		if detector.start_reid and DATASET == 'DAVIS':
			iou = eval.compute_IoU(cv2.cvtColor(np.float32(seg), cv2.COLOR_BGR2GRAY) > 0, eval.get_gt_mask(image))
			print("Intersection over Union wrt. ground truth: ", iou)
			ious[image] = iou
			
		show_img = img.copy()
		ui.plot_boxes(show_img, prob, boxes)
		cv2.imshow('image', show_img)
		
		k = cv2.waitKey(int(1000/fps)) & 0xFF ## TODO: find alternative to waitKey such that operations can be performed in parallel
		if k == 27:
			break
	
	print("Mean IoU: ", np.mean(list(ious.values())))


	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()