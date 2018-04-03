from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import numpy as np
import cv2
import dlib
import sys
from cloth_detector import use_cloth_neural_network
from face_detector import use_face_neural_network
#########################################################################
		### Initialisation of detectors
detector = dlib.get_frontal_face_detector()
desc = cv2.HOGDescriptor()
desc.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
###########################################################################
		### Detecting Pedestrian in images
image = cv2.imread(sys.argv[1])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = imutils.resize(image, width=min(400, image.shape[1]))
(rects, weights) = desc.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
ped_dect = non_max_suppression(rects, probs=None, overlapThresh=0.65)
#############################################################################
		### Considering the cropped part containing pedestrian
print len(ped_dect) 
for (x1, y1, x2, y2) in ped_dect:
	image2 = image[y1:y2,x1:x2]
#############################################################################
		### Gender classifier
	dets = detector(image2, 1)
	face = image2[dets[0].top():dets[0].bottom(),dets[0].left():dets[0].right()]
	face = cv2.resize(face,(128,128), interpolation = cv2.INTER_AREA)
	face = np.array(face,dtype='float32')
	normalizedImg = np.zeros((128, 128),dtype='float32')	##Normalising the image 0-1
	normalizedImg = cv2.normalize(face,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
	flat_img = np.reshape(normalizedImg,(1,16384))
	result = use_face_neural_network(flat_img)
	print result
#################################################################################
	body = cv2.resize(image2,(128,128), interpolation = cv2.INTER_AREA)
	body = np.array(body,dtype='float32')
	normalizedImg = np.zeros((128, 128),dtype='float32')	##Normalising the image 0-1
	normalizedImg = cv2.normalize(body,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
	flat_img = np.reshape(normalizedImg,(1,16384))
	result = use_cloth_neural_network(flat_img)
	print result

