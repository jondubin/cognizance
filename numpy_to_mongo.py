##
import os
import cv2
successes = 0
failures = 0
kp_list = []
des_list = []
logo_directory = '../scraper/logos'
for file in os.listdir(logo_directory):
	try:
		print("Reading " + logo_directory + "/" + file)
		img = cv2.imread(logo_directory + "/" + file, 0)
		surf = cv2.SURF(400)
		kp, des = surf.detectAndCompute(img,None)
		kp_list.append(kp)
		des_list.append(des)
		successes += 1
	except:
		failures += 1
	print("  Successes: " + str(successes))
	print("  Failures:  " + str(failures))

import cPickle
from pymongo import MongoClient
from bson.binary import Binary
conn = MongoClient()


##Create 3 collections: one for your dataset, one for dubin's dataset, and one wih both combined.


collection = conn.logos.descriptors
for des in des_list:
	collection.insert({'cpickle': Binary(cPickle.dumps(des, protocol=2))})