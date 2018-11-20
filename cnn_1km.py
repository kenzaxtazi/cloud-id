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
import matplotlib.pyplot as plt
import DataLoader as dl
import re
from PrepData import prep_data
import Collocation as coll
from tflearn.layers.conv import conv_2d, max_pool_2d

# directory where we are storing the 1km CALIOP data

directory= '/Users/kenzatazi/Downloads/Directory'
LR = 1e-3

 
MODEL_NAME = 'cnn_1km'.format(LR, 'convolutional') 


###### PREPROCESSING

match_pathnames= prep_data.open_matches()

caliop_directory= '/home/hep/trz15/cloud/Calipso/1km/2018/04'
slstr_directory= '/home/hep/trz15/cloud/SLSTR/2018/04'


CALIOP_pathnames= []
SLSTR_pathnames= []

# Please fill in below:
for p in match_pathnames:
    print((p[0])[43:45])
    CALIOP_pathnames.append(caliop_directory+(p[0])[43:45]+'/'+p[0])
    SLSTR_pathnames.append(slstr_directory + p[1])

# collcates pixels returns [SLSTR_row, SLSTR_col, CALIPSO_index]
pixels=[]
for n in range(CALIOP_pathnames):
    pixels.extend(coll.collocate(SLSTR_pathnames[n], CALIOP_pathnames[n]))

pixel_info = prep_data.save_data(pixels, SLSTR_pathnames, CALIOP_pathnames)

training_data, validation_data, training_truth, validation_truth = prep_data(pixel_info)

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


model.fit(training_data, training_truth, n_epoch=1, validation_set =
          (validation_data, validation_truth), snapshot_step=1000, 
          show_metric=True, run_id=MODEL_NAME)


reset_default_graph()
