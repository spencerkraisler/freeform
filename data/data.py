import numpy as np 
import cv2


i = 0
for i in range(300):
	image = np.load("full-numpy_bitmap-golf club.npy")[i]
	image = np.reshape(image, (28, 28))
	cv2.imwrite("./golf_clubs/IMG_" + str(i) + ".jpg", image)

