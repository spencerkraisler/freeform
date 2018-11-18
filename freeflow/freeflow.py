import cv2
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
from backproj import mask, getHistogram
from contours import getContours, handleContours

centers = []

def getContourCenter(contours, frame=None, draw_center=False):
	if len(contours) > 0:
		center = (0,0)
		for cnt in contours:
			(x,y,w,h) = cv2.boundingRect(cnt)
			center = (x + w//2,y + h//2)
			if draw_center:
				cv2.circle(frame, center, 2, (0, 255, 0), 3)
		return center


def getLength(p1, p2):
	return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def drawCenters(centers, frame):
	size = len(centers)
	for i in range(size):
		c = centers[i]
		if i > 0:
			c_last = centers[i-1]
			if c_last != None and c != None and getLength(c_last, c) < 100:
					cv2.line(frame, c_last, c, (0, 255, 0), 4)
	
def handleCenters(centers):
	if len(centers) > 400:
		del centers[0]	

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
		centers.append(getContourCenter(contours, frame, draw_center=True))
		drawCenters(centers, frame)
		frame = np.flip(frame, 1)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


roi_img = cv2.imread('./images/roi.jpg', 3)
roi_hist = getHistogram(roi_img)

startVideoFeed(0)
