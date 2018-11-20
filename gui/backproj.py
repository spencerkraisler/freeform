import cv2
import numpy as np 


def mask(hists, img):
	img = cv2.blur(img, (5, 5))
	mask = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	mask1 = cv2.calcBackProject([mask], [1, 2], hists[0], [0, 255, 0, 255], 1)
	mask2 = cv2.calcBackProject([mask], [1, 2], hists[1], [0, 255, 0, 255], 1)
	mask = ((mask1 + mask2) / 2).astype('uint8')
	return mask

def getHistogram(img):
	lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	hist = cv2.calcHist([lab_img], [1, 2], None, 
		 				[15, 15], [0, 255, 0, 255])
	cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
	return hist