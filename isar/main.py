import numpy as np
import cv2
import os
from ui.ui import UserInterface
from detector import Detector
from reidentification import Reidentification
from util.isar_utils import get_image_it_from_folder

import sys
sys.path.append('external/dino-vit-features')


datadir = "/home/nico/semesterproject/data/videos/3dod/Training/47670305/47670305_frames/lowres_wide/"
fps = 15


def main():
	detector = Detector()
	ui = UserInterface(detector=detector)

	# ui.play_from_folder()
	images = get_image_it_from_folder(datadir, fps)

	while True:
		img = images.__next__()
		ui.img = img
		prob, boxes = detector.detect(img)
		show_img = img
		# ui.plot_boxes(show_img, prob, boxes)
		cv2.imshow('image', show_img)
		k = cv2.waitKey(int(1000/fps)) & 0xFF ## TODO: find alternative to waitKey such that operations can be performed in parallel
		if k == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()