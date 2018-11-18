import cv2
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
from backproj import mask, getHistogram
from contours import getContours, handleContours

centers = []

def getContourCenter(contours):
	center = (0,0)
	for cnt in contours:
		(x,y,w,h) = cv2.boundingRect(cnt)
		center = (x+w//2,y+h//2)
	return center

def drawCenters(frame, centers):
	for i in range(len(centers)):
		c = centers[i]
		cv2.circle(frame, c, 4, (0, 255, 0), 3)

def startVideoFeed(cam_index, hist=None):

	# Args:
		# cam_index (int): 0 for webcam; 1 for USB camera
		# hist (numpy array): histogram for masking
	cap = cv2.VideoCapture(cam_index)
	while(True):
		_, frame = cap.read()
		thresh_frame = mask(roi_hist, frame)
		contours = getContours(thresh_frame)
		contours = handleContours(contours)
		centers.append(getContourCenter(contours))
		drawCenters(frame, centers)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


roi_img = cv2.imread('./images/roi.jpg', 3)
roi_hist = getHistogram(roi_img)

startVideoFeed(0)
