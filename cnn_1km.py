#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:02:38 2018

@author: kenzatazi
"""

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
import DataLoader as dl
import re

import Plotting_images_and_masks as pim

import vfm_feature_flags as vfm 
import CalipsoReader2 as cr2
import Collocation as coll
from pyhdf.SD import SD, SDC

from tflearn.layers.conv import conv_2d, max_pool_2d

# directory where we are storing the 1km CALIOP data

directory= '/Users/kenzatazi/Downloads/Directory'
LR = 1e-3

 
MODEL_NAME = 'multi_layer_perceptron'.format(LR, 'convolutional') 


###### PREPROCESSING


# Please fill in below:

SLSTR_pathname =
CALIOP_pathname =

# collcates pixels
pixels= coll.collocate(SLSTR_pathname, CALIOP_pathname) # [SLSTR_row, SLSTR_col, CALIPSO_index]



def prep_data(pixels, SLSTR_pathname,CALIOP_pathname):
    
    """ prepare data for one CALIOP and one SLSTR file at a time """
    
    #Load SLSTR file 
    scn = Scene(filenames=SLSTR_pathname, reader='nc_slstr')
    pim.load_scene(scn)
        
    #Load CALIOP file 
    file = SD(CALIOP_pathname, SDC.READ)
    data=cr2.load_data(CALIOP_pathname,'Feature_Classification_Flags')
    
    pixel_info=[]

    for val in pixels:
        
        y,x,v = val[0],val[1],val[2]  
        
        truth_set=[]
        
        if 1 < float(vfm.vfm_feature_flags(data[v,0])) < 4:
            truth_set.append(np.array([0,1])) # cloud 
        else:
            truth_set.append(np.array([1,0])) # not cloud
        
        S1set=[]
        S2set=[]
        S3set=[]
        S4set=[]
        S5set=[]
        S6set=[]
        S7set=[]
        S8set=[]
        S9set=[]
                
        for i in range(-4,5,2):
            for j in range(-4,5,2):
                
                S1set.extend([float((scn['S1_an'])[x+i,y+j]+
                                    (scn['S1_an'])[x+i+1,y+j]+
                                    (scn['S1_an'])[x+i,y+1+j]+
                                    (scn['S1_an'])[x+i+1,y+j+1])/4.])
                S2set.extend([float((scn['S2_an'])[x+i,y+j]+
                                    (scn['S2_an'])[x+i+1,y+j]+
                                    (scn['S2_an'])[x+i,y+1+j]+
                                    (scn['S2_an'])[x+i+1,y+j+1])/4.])
                S3set.extend([float((scn['S3_an'])[x+i,y+j]+
                                    (scn['S3_an'])[x+i+1,y+i]+
                                    (scn['S3_an'])[x+i,y+1+j]+
                                    (scn['S3_an'])[x+i+1,y+j+1])/4.])
                S4set.extend([float((scn['S4_an'])[x+i,y+j]+
                                    (scn['S4_an'])[x+i+1,y+j]+
                                    (scn['S4_an'])[x+i,y+1+j]+
                                    (scn['S4_an'])[x+i+1,y+j+1])/4.])
                S5set.extend([float((scn['S5_an'])[x+i,y+j]+
                                    (scn['S5_an'])[x+i+1,y+j]+
                                    (scn['S5_an'])[x+i,y+1+j]+
                                    (scn['S5_an'])[x+i+1,y+i+1])/4.])
                S6set.extend([float((scn['S6_an'])[x+i,y+j]+
                                    (scn['S6_an'])[x+i+1,y+j]+
                                    (scn['S6_an'])[x+i,y+1+j]+
                                    (scn['S6_an'])[x+i+1,y+j+1])/4.])
                S7set.extend([(scn['S7_in'])[int(float(x+i)/2.),int(float(y+j)/2.)]]) 
                S8set.extend([(scn['S8_in'])[int(float(x+i)/2.),int(float(y+j)/2.)]])
                S9set.extend([(scn['S9_in'])[int(float(x+i)/2.),int(float(y+j)/2.)]])
                
        pixel_info.append([S1set, S2set, S3set, S4set, S5set, S6set,
                            S7set, S8set, S9set, truth_set]) 
    
    np.nan_to_num(pixel_info)
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
    



training_data, validation_data, training_truth, validation_truth = prep_data(pixels, )

training_data= training_data.reshape(-1,5,5,9)
validation_data= validation_data.reshape(-1,5,5,9)


#### MACHINE LEARNING 



import tflearn 
from tensorflow import reset_default_graph
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Layer 0: generates a 4D tensor
layer0 = input_data(shape=[None, 5, 5, 9], name='input')

# Layer 1
convnet = conv_2d(layer0, nb_filter=32, filter_size=4, activation='relu')
convnet = max_pool_2d(convnet,kernel_size=5)

# Layer 2
convnet = conv_2d(convnet, nb_filter=64, filter_size=4, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 3
convnet = conv_2d(convnet, nb_filter=128, filter_size=4, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 4
convnet = conv_2d(convnet, nb_filter=64, filter_size=4, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 5
convnet = conv_2d(convnet, nb_filter=32, filter_size=4, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 6
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, keep_prob=0.8)

# Layer 7 (this layer needs to spit out the number of categories we are looking for)
convnet = fully_connected(convnet, n_units=2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, \
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_verbose=0)



### UNPACK SAVED DATA


model.fit(training_data, training_truth, n_epoch=1, validation_set =
          (validation_data, validation_truth), snapshot_step=1000, 
          show_metric=True, run_id=MODEL_NAME)


reset_default_graph()
