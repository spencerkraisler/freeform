import cv2

def getHullDefects(contours):
	defects = []
	for cnt in contours:
		hull = cv2.convexHull(cnt, returnPoints=False)
		defects.append(cv2.convexityDefects(cnt, hull))
	return defects

def drawConvexHulls(img, contours, hulls):
	for i in range(len(contours)):
		cnt = contours[i]
		hull = hulls[i]
		for i in range(hull.shape[0]):
			s,e,f,d = hull[i,0]
			start = tuple(cnt[s][0])
			end = tuple(cnt[e][0])
			far = tuple(cnt[f][0])
			cv2.line(img, start,end,[0,255,0],2)
			cv2.circle(img,far,5,[0,0,255],-1)
