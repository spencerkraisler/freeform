import cv2

def mask(hist, img):
	img = cv2.blur(img, (5, 5))
	mask = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	mask = cv2.calcBackProject([mask], [1, 2], hist, [0, 255, 0, 255], 1)
	#mask = denoise(mask)
	#mask = cv2.blur(mask, (5,5))
	return mask

def getHistogram(img):
	lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	hist = cv2.calcHist([lab_img], [1, 2], None, 
		 				[15, 15], [0, 255, 0, 255])
	cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
	#hist = dilate(hist, DILATION_KERNEL_SIZE)
	return hist