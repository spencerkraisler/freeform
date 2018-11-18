import numpy as np 
import cv2


i = 0
for i in range(100):
	image = np.load("./full-numpy_bitmap-bicycle.npy")[i + 8600]
	image = np.reshape(image, (28, 28))
	cv2.imwrite("./images/bicycles/IMG_" + str(i + 600) + ".jpg", image)

