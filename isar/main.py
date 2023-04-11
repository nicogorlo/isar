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


#AR-KIT:
# dataset = 'AR-KIT'
# task = "47670305"
# datadir = "/home/nico/semesterproject/data/ARKit_data/3dod/Training/"
# img_dir = f'{task}/{task}_frames/lowres_wide/'
# eval_dir = None

#Cityscapes:
dataset = 'Cityscapes'
task = 'stuttgart_00'
datadir = "/home/nico/semesterproject/data/Cityscapes/"
img_dir = os.path.join('leftImg8bit_demoVideo/leftImg8bit/demoVideo/', task)
eval_dir = None

#DAVIS:
# dataset = 'DAVIS'
# task = "breakdance"
# datadir = "/home/nico/semesterproject/data/DAVIS-2017-Unsupervised-trainval-480p/DAVIS/"
# img_dir = os.path.join("JPEGImages/480p/", task)
# eval_dir = os.path.join("Annotations_unsupervised/480p/", task)
# out_dir = os.path.join("home/nico/semesterproject/test/", task)


fps = 15


def main():
	detector = Detector()
	ui = UserInterface(detector=detector)
	if dataset == 'DAVIS':
		eval = Evaluation(datadir, eval_dir, dataset)

	ious = {}

	# ui.play_from_folder()
	images = get_image_it_from_folder(os.path.join(datadir,img_dir), fps)

	while True:
		image = images.__next__()
		img = cv2.imread(os.path.join(datadir, img_dir, image))
		# img = cv2.resize(img, (285,160))
		ui.img = img
		prob, boxes, seg = detector.detect(img, image)

		if detector.start_reid and dataset == 'DAVIS':
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