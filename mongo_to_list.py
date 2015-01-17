import cPickle
from pymongo import MongoClient
from bson.binary import Binary

##This returns the arrays

conn = MongoClient()
collection = conn.logos.descriptors

larray = [cPickle.loads(x['cpickle']) for x in collection.find()]