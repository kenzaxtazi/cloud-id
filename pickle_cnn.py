#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:37:26 2018

@author: kenzatazi
"""
# open pickled file

import pandas as pd
import numpy as np
from random import shuffle
import vfm_feature_flags2 as vfm 

LR = 1e-3
MODEL_NAME = 'pickle_cnn'.format(LR, 'convolutional') 

pixel_info = pd.read_pickle("/Users/kenzatazi/Desktop/AprilP1.pkl")
pixel_values = (pixel_info[['S1_an','S2_an','S3_an','S4_an','S5_an','S6_an',
                           'S7_in','S8_in','S9_in', 
                           'Feature_Classification_Flags']]).values



def prep_data(pixel_info):
    
    """ 
    Prepares data for matched SLSTR and CALIOP pixels into training data, 
    validation data, training truth data, validation truth data.
    
    """
    
    shuffle(pixel_info)     # mix real good
    
    data = pixel_info[:,:-1]
    truth_flags = pixel_info[:,-1]
    
    truth_oh=[]
    
    for d in truth_flags:
        i = vfm.vfm_feature_flags(int(d))
        if i == 2:
            truth_oh.append([1,0])    # cloud 
        if i != 2:
            truth_oh.append([0,1])    # not cloud 
        
  
    training_data= np.array(data[:-500])      # take all but the 500 last 
    validation_data= np.array(data[-500:])    # take 500 last pixels 
    training_truth= np.array(truth_oh[:-500])
    validation_truth= np.array(truth_oh[-500:])
    
    return training_data, validation_data, training_truth, validation_truth
        


# prepares data for cnn 

training_data, validation_data, training_truth, validation_truth = prep_data(pixel_values)



#### MACHINE LEARNING 

import tflearn 
from tensorflow import reset_default_graph
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d

training_data= training_data.reshape(-1,1,9,1)
validation_data= validation_data.reshape(-1,1,9,1)

# Layer 0: generates a 4D tensor
layer0 = input_data(shape=[None, 1, 9, 1], name='input')

# Layer 1
layer1 = fully_connected(layer0, 32, activation='relu')
dropout1 = dropout(layer1,0.8) ## what is dropout?

# Layer 2
layer2 = fully_connected(dropout1, 32, activation='relu')
dropout2 = dropout(layer2,0.8)

# Layer 3
layer3 = fully_connected(dropout2, 32, activation='relu')
dropout3 = dropout(layer3,0.8)

# Layer 4
layer4 = fully_connected(dropout3, 32, activation='relu')
dropout4 = dropout(layer4,0.8)

#this layer needs to spit out the number of categories we are looking for.
softmax = fully_connected(dropout4, 2, activation='softmax') 


network = regression(softmax, optimizer='adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(network, tensorboard_verbose=0)



### UNPACK SAVED DATA


model.fit(training_data, training_truth, n_epoch=1, validation_set =
          (validation_data, validation_truth), snapshot_step=1000, 
          show_metric=True, run_id=MODEL_NAME)


reset_default_graph()