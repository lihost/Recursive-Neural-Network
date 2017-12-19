#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:36:49 2017

Flickr8k dataset has been used to prepare sentence embeddings.

@author: sohil
"""

#generate Word2vec vectors for sentences
from gensim.models import Word2Vec
import logging
import os
import numpy as np

#Ignore all words with total frequency lower than this
MIN_COUNT = 1
#Dimensionality
SIZE = 50
#number of worker threads to use while training i.e. number of cores in machine
WORKERS = 4
#initial learning rate
ALPHA = 0.05
#cbow_mean
#CBOW_MEAN = 1

SENT_DIR = '../data/raw/'
SAVE_MODEL = '../data/w2v/test_sent5_We'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#sentence iterator class
class SentenceIter(object):
    def __init__(self, dirname):
        self.dirname = dirname
        
    def __iter__(self):
        for fName in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fName)):
                yield line.split()


#loading sentences
sentences = SentenceIter(SENT_DIR)
model = Word2Vec(sentences, size=SIZE, min_count=MIN_COUNT, workers = WORKERS, alpha=ALPHA)

#saving model
#model.save(SAVE_MODEL)
#model.wv.save_word2vec_format(SAVE_MODEL)
weights = model.wv.syn0
np.save(SAVE_MODEL, weights)


#vocab = list(model.wv.vocab.keys())
#print(vocab[:10])

#print(model.most_similar('person'))

