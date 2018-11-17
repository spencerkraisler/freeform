import cv2
import numpy as np 
import os 

path = "./open/"
files = os.listdir(path)
for file in files:
	image = cv2.imread(path + file)
	cv2.imwrite("./aux/" + file[8:], image)