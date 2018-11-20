import tensorflow as tf 
import numpy as np 
import cv2
from math import sqrt 

categories = {1: "index"}

def load_graph(frozen_graph_filename):
	with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name='prefix')
	return graph

def drawCenters(frame):
	size = len(centers)
	for i in range(size):
		if i > 0:
			c = centers[i]
			c_last = centers[i - 1]
			if c != None and c_last != None:
				cv2.line(frame, c_last, c, (0, 255, 0), 4)

def denormalize_bbox(bbox):
	new_bbox = np.zeros(bbox.shape)
	new_bbox[0] = bbox[0] * 720
	new_bbox[1] = bbox[1] * 1280
	new_bbox[2] = bbox[2] * 720
	new_bbox[3] = bbox[3] * 1280
	new_bbox = new_bbox.astype(np.int32)
	return new_bbox

GRAPH_PATH = "./checkpoints/frozen_inference_graph.pb"
NUM_CLASSES = 1
detection_graph = load_graph(GRAPH_PATH)
SCORE_THRESHOLD = 98.5

centers = []

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		cap = cv2.VideoCapture(0)
		while(True):
			_, frame = cap.read()
			frame_exp = np.expand_dims(frame, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('prefix/image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('prefix/detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('prefix/detection_scores:0')
			classes = detection_graph.get_tensor_by_name('prefix/detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('prefix/num_detections:0')
			
			(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections],
				feed_dict={image_tensor: frame_exp})

			box = denormalize_bbox(boxes[0][0])
			

			if (scores[0][0] * 100 > SCORE_THRESHOLD):
				x = box[1] + 50
				y = box[0] + 50
				centers.append((x,y))
				#cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 3)
				print(scores[0][0] * 100)
			drawCenters(frame)
				
			cv2.imshow('frame', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
