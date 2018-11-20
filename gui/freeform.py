import cv2
from math import sqrt
import numpy as np 
from backproj import mask, getHistogram
from contours import getContours, handleContours
from PIL import Image 

centers = []
pixels = []
beta = .6

SCREEN_HEIGHT = 800
SCREEN_WIDTH = 1280

FRAME_HEIGHT = 600
FRAME_WIDTH = 800

CLEAR_MIN = (800, 50)
CLEAR_MAX = (1200, 150)

def drawButton(canvas):
	canvas = cv2.rectangle(canvas, CLEAR_MIN, CLEAR_MAX, (0, 0, 255), -1)
	cv2.putText(canvas, "CLEAR ALL", (CLEAR_MIN[0] + 40, CLEAR_MAX[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
	

def buttonCheck(min_coords, max_coords, x, y):
	print(x,y)
	if x in range(CLEAR_MIN[0], CLEAR_MAX[0]) and y in range(CLEAR_MIN[1], CLEAR_MAX[1]):
		print("CLEARR")
		return True
	else: return False

def getContourCenter(contours, frame=None, draw_center=False):
	if len(contours) > 0:
		center = (0,0)
		cnt = contours[0]
		M = cv2.moments(cnt)
		(x,y) = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
		center = (x,y)
		if draw_center:
			if x == None:
				cv2.circle(frame, centers[-1], 2, (0, 255, 0), 3)
			else:
				if len(pixels) == 0:
					pixels.append(center)
				pixel_x = int(beta * pixels[-1][0] + (1 - beta) * center[0])
				pixel_y = int(beta * pixels[-1][1] + (1 - beta) * center[1])
				pixels.append((pixel_x, pixel_y))
				cv2.circle(frame, pixels[-1], 30, (0, 0, 0), 3)
		return center


def getLength(p1, p2):
	return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def drawCenters(pixels, frame):
	size = len(pixels)
	for i in range(size):
		if i > 0:
			p = pixels[i]
			p_last = pixels[i - 1]
			if p_last != None and p != None and getLength(p_last, p) < 200:
					cv2.line(frame, p_last, p, (0, 255, 0), 4)

def startVideoFeed(cam_index, hist=None):

	# Args:
		# cam_index (int): 0 for webcam; 1 for USB camera
		# hist (numpy array): histogram for masking
	cap = cv2.VideoCapture(cam_index)
	while(True):
		_, frame = cap.read()
		thresh_frame = mask((hist1, hist2), frame)
		contours = getContours(thresh_frame)
		contours = handleContours(contours)
		canvas = np.ones(frame.shape) * 255
		
		centers.append(getContourCenter(contours, canvas, draw_center=True))
		drawCenters(pixels, canvas)
		canvas = np.flip(canvas, 1).copy()
		drawButton(canvas)
		if len(pixels) > 0:
			if buttonCheck(CLEAR_MIN, CLEAR_MAX, SCREEN_WIDTH - pixels[-1][0], pixels[-1][1]):
				canvas = np.ones(frame.shape) * 255
				drawButton(canvas)
				centers.clear()
				pixels.clear()
				print("CLEAR")

		canvas_resized = cv2.resize(canvas, (FRAME_WIDTH, FRAME_HEIGHT))
		cv2.imshow('canvas', canvas_resized)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

roi1 = cv2.imread('./images/roi_red.jpg', 3)
roi2 = cv2.imread('./images/roi_red.jpg', 3)
hist1 = getHistogram(roi1)
hist2 = getHistogram(roi2)


startVideoFeed(0)
