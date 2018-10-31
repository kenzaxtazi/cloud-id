#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:12:12 2018

@author: kenzatazi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:28:06 2018

@author: kenzatazi
"""


import os
import numpy as np
from glob import glob
from satpy import Scene
from random import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm   #percentage bar for tasks.


directory= '/Users/kenzatazi/Downloads/Directory'
LR = 1e-3

 
MODEL_NAME = 'multi_layer_perceptron'.format(LR, 'feed-foward') 


def load_scene(scn):
    """ Loads the information from the netcdf files in the folder"""
    scn.load(['S1_an','S2_an','S3_an','S4_an','S5_an','S6_an',
              'bayes_an','cloud_an'])


def create_mask(scn, mask_name):
    """Extracts bitmasks and combines them into an overall mask array"""
    mask=[]
    for bitmask in scn[mask_name].flag_masks[:-2]:
        data = scn[mask_name].values & bitmask
        mask.append(data)
    mask= np.sum(mask, axis=0)
    return mask


###### PREPROCESSING

    
def prep_data(directory):
    """ 
    Dowloads the data and puts it into a usable format
    Returns training and validation data sets as well as truth sets. 
    """ 
    
    pixel_info=[]
    
    
    for folder in tqdm(os.listdir(directory)):
        path = os.path.join(directory,folder,'*')
        print(path)
        scn = Scene(filenames=glob(path), reader='nc_slstr')
        
        load_scene(scn)
        
        S1= np.nan_to_num(scn['S1_an'].values)[0:200,0:200]
        S2= np.nan_to_num(scn['S2_an'].values)[0:200,0:200]
        S3= np.nan_to_num(scn['S3_an'].values)[0:200,0:200]
        S4= np.nan_to_num(scn['S4_an'].values)[0:200,0:200]
        S5= np.nan_to_num(scn['S5_an'].values)[0:200,0:200]
        S6= np.nan_to_num(scn['S6_an'].values)[0:200,0:200]
        
        
        bayes_mask= create_mask(scn, 'cloud_an')
        bayes_mask=np.array(bayes_mask)[:200,:200]
        
        bayes_mask=(np.reshape(bayes_mask,(1,-1))[0])
        
        #one hot encoding for mask
        truth_set=[]
        
        for i in tqdm(bayes_mask):
            if i>0:
                truth_set.append(np.array([0,1]))
            else:
                truth_set.append(np.array([1,0]))
            
        for x in range(200):
            print(x)
            for y in range(200):
                pixel_info.append([S1[x,y], S2[x,y], S3[x,y], S4[x,y], S5[x,y],
                                   S6[x,y], truth_set[y+y*(x-1)]]) 
    
    
    shuffle(pixel_info)     # mix real good

    data= []
    truth= []
    
    for i in pixel_info:
        data.append(i[:-1])
        truth.append(i[-1])
        
  
    training_data= np.array(data[:-500]) # take all but the 500 last 
    validation_data= np.array(data[-500:])    # take 500 last pixels 
    training_truth= np.array(truth[:-500])
    validation_truth= np.array(truth[-500:])
    
    
    return training_data, validation_data, training_truth, validation_truth


training_data, validation_data, training_truth, validation_truth = prep_data(directory)

training_data= training_data.reshape(-1,1,6,1)
validation_data= validation_data.reshape(-1,1,6,1)


#### MACHINE LEARNING 



import tflearn 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Layer 0: generates a 4D tensor
layer0 = input_data(shape=[None, 1, 6, 1], name='input')

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


model.fit(training_data, training_truth, n_epoch=5, validation_set =
          (validation_data, validation_truth), snapshot_step=500, 
          show_metric=True, run_id=MODEL_NAME)

