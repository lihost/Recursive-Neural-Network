#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:20:21 2017

@author: sohil
"""

import numpy as np
import max_margin_dtr


DEFAULT_IMAGE_DIM = 2048
DEFAULT_SENT_DIM = 50
    
def prepare_datalist(sent, image_vec, sent_vec):
    data_list = []
    sents = []
    with open(sent, 'r') as f:
	    for lines in f:
		    sents.append(lines.rstrip())

    #sentence list added
    data_list.append(sents)

    img_data = np.load(image_vec)
    #reshaping to 2-dim
    #i_d = img_data.reshape(1, len(img_data), DEFAULT_IMAGE_DIM).swapaxes(1,2).reshape(len(img_data), -1)
    #reshaping it properly
    i_d = img_data.reshape(-1, img_data.shape[-1])
    data_list.append(i_d)


    s = np.load(sent_vec)
    sent_vec = s.astype(np.float32, copy=False)
    #reshaping to 2-dim
    sent_vec = sent_vec.reshape(len(s),DEFAULT_SENT_DIM)
    data_list.append(sent_vec)
    return data_list


if __name__ == '__main__':
    
    
    train_sent = 'data/train_sent5.txt'
    train_image_vec = 'data/img_vec_train5.npy'
    train_sent_vec = 'data/sent_vec_train5.npy'

    test_sent = 'data/test_sent5.txt'
    test_image_vec = 'data/img_vec_test5.npy'
    test_sent_vec = 'data/sent_vec_test5.npy'

    dev_sent = 'data/dev_sent5.txt'
    dev_image_vec = 'data/img_vec_dev5.npy'
    dev_sent_vec = 'data/sent_vec_dev5.npy'

    train_list = prepare_datalist(train_sent, train_image_vec, train_sent_vec)
    test_list = prepare_datalist(test_sent, test_image_vec, test_sent_vec)
    dev_list = prepare_datalist(dev_sent, dev_image_vec, dev_sent_vec)

    #For Training
    max_margin_dtr.trainer(train_list, test_list)

    #For evaluation, uncomment it.
    #max_margin_dtr.evaluate(test_list, 'saved_max_margin_model.npz', evaluate=True)
