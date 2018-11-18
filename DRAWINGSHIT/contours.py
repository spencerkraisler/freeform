import cv2
import numpy as np 



def getContours(mask):
	_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours

THRESH_CONTOUR_AREA = 2000

def threshold_area(contours):
	out_contours = []
	for cnt in contours:
		if cv2.contourArea(cnt) >= THRESH_CONTOUR_AREA:
			out_contours.append(cnt)
	return out_contours

def handleContours(contours):
	contours = threshold_area(contours)
	return contours