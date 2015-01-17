##
import cv2
img = cv2.imread('./logo.png',0)
surf = cv2.SURF(400)
kp, des = surf.detectAndCompute(img,None)

import cPickle
from pymongo import MongoClient
from bson.binary import Binary
conn = MongoClient()


##Create 3 collections: one for your dataset, one for dubin's dataset, and one wih both combined.


collection = conn.logos.descriptors
collection.insert({'cpickle': Binary(cPickle.dumps(des, protocol=2))})

##This returns the arrays 
test = [cPickle.loads(x['cpickle']) for x in collection.find()]