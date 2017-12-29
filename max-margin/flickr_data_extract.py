#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:20:21 2017

@author: sohil
"""
import os
import operator

FLICKR8K_DIR = "~Documents/Flickr8k_text/"
TRAIN_SENT_FILE = "train_sent5.txt"
TEST_SENT_FILE = "test_sent5.txt"
DEV_SENT_FILE = "dev_sent5.txt"

def flickrXYFilenames(n_examples=None, dataType='8k'):
    if dataType == '8k':
        FLICKR_DIR = FLICKR8K_DIR
    else:
        FLICKR_DIR = FLICKR30K_DIR
    feature_path = os.path.join(FLICKR_DIR, "features")
    caption_path = os.path.join(FLICKR_DIR, "Flickr8k.token.txt")

    f = open(caption_path, 'r')
    lines = f.read().splitlines()
    
    data_dict = {}

    def parse(line):
        split = line.split('#')
        img = split[0]
        caption_ugly = split[1:]
        if isinstance(caption_ugly, list):
            caption_ugly = ' '.join(caption_ugly)
        i, caption = caption_ugly.split('\t')
        handful = data_dict.get(img, [])
        handful.append(caption)
        data_dict[img] = handful

    print("Parsing flickr%s captions..." % dataType)
    for i, line in enumerate(lines):
        parse(line)

    #Y=sentences, fns=images	
    fns, Y = zip(*data_dict.items())
	
    return Y, fns


def prepare_dev_content(fp):
    Y, fns = flickrXYFilenames()
    #print(Y)
    #print(fns)
    s = []
    with open(fp, 'r') as f:
        for l in f.readlines():
            i = fns.index(l.strip())
	    #s.append(Y[i][0].strip(' .')+'.\n')
	    t = [x.strip(' .')+'.\n' for x in Y[i]]
	    s.append(reduce(operator.concat, t))
	    #print(s)	
    s = ''.join(str(p) for p in s)
    #s = ''.join(s)
    #print(s)
    return s

def create_file(fp, s):
    with open(fp, 'w') as f:
        f.write(s)



if __name__ == '__main__':
    
    train_path = os.path.join(FLICKR8K_DIR, 'Flickr_8k.trainImages.txt')
    test_path = os.path.join(FLICKR8K_DIR, 'Flickr_8k.testImages.txt')
    dev_path = os.path.join(FLICKR8K_DIR, 'Flickr_8k.devImages.txt')
    
    s_train = prepare_dev_content(train_path)
    s_test = prepare_dev_content(test_path)
    s_dev = prepare_dev_content(dev_path)
    #print(s_dev)
    
    print('writing training sentences to: %s file' %(TRAIN_SENT_FILE))
    create_file(TRAIN_SENT_FILE, s_train)
    
    print('writing test sentences to: %s file' %(TEST_SENT_FILE))
    create_file(TEST_SENT_FILE, s_test)

    print('writing dev sentences to: %s file' %(DEV_SENT_FILE))
    create_file(DEV_SENT_FILE, s_dev)
    
                
                
        
