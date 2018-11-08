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
from tflearn.layers.conv import conv_2d, max_pool_2d

directory= '/Users/kenzatazi/Downloads/Directory'
LR = 1e-3

 
MODEL_NAME = 'multi_layer_perceptron'.format(LR, 'feed-foward') 


def load_scene(scn):
    """ Loads the information from the netcdf files in the folder"""
    scn.load(['S1_an','S2_an','S3_an','S4_an','S5_an','S6_an','S7_in','S8_in',
              'S9_in','bayes_an', 'bayes_in','cloud_an'])


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
    
    olddir = os.getcwd()
    pixel_info=[]
    os.chdir(directory)
    fpattern = dl._regpatternf()
    
    
    for folder in tqdm(os.listdir()):
        if re.findall(fpattern, folder) != []:
            filenames = glob(str(folder) + "/*")
            filenames = dl.fixdir(filenames)
            scn = Scene(filenames=filenames, reader='nc_slstr')
            
            load_scene(scn)
            
            S1= np.nan_to_num(scn['S1_an'].values)
            S2= np.nan_to_num(scn['S2_an'].values)
            S3= np.nan_to_num(scn['S3_an'].values)
            S4= np.nan_to_num(scn['S4_an'].values)
            S5= np.nan_to_num(scn['S5_an'].values)
            S6= np.nan_to_num(scn['S6_an'].values)
            S7= np.nan_to_num(scn['S7_in'].values)
            S8= np.nan_to_num(scn['S8_in'].values)
            S9= np.nan_to_num(scn['S9_in'].values)
            
            mask= create_mask(scn, 'bayes_in')
            mask=np.array(mask)
            
            mask=(np.reshape(mask,(1,-1))[0])
            
            #one hot encoding for mask
            truth_set=[]
            
            for i in tqdm(mask):
                if i>0:
                    truth_set.append(np.array([0,1]))
                else:
                    truth_set.append(np.array([1,0]))
            
            pixel_info=[]
            
            
            for x in range(6,len(S1)-6,10):
                for y in range(6,len(S1)-6,10):
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
                            
                            S1set.extend([float(S1[x+i,y+j]+S1[x+i+1,y+j]+
                                                S1[x+i,y+1+j]+S1[x+i+1,y+j+1])/4.])
                            S2set.extend([float(S2[x+i,y+j]+S2[x+i+1,y+j]+
                                                S2[x+i,y+1+j]+S2[x+i+1,y+j+1])/4.])
                            S3set.extend([float(S3[x+i,y+j]+S3[x+i+1,y+i]+
                                                S3[x+i,y+1+j]+S3[x+i+1,y+j+1])/4.])
                            S4set.extend([float(S4[x+i,y+j]+S4[x+i+1,y+j]+
                                                S4[x+i,y+1+j]+S4[x+i+1,y+j+1])/4.])
                            S5set.extend([float(S5[x+i,y+j]+S5[x+i+1,y+j]+
                                                S5[x+i,y+1+j]+S5[x+i+1,y+i+1])/4.])
                            S6set.extend([float(S6[x+i,y+j]+S6[x+i+1,y+j]+
                                                S6[x+i,y+1+j]+S6[x+i+1,y+j+1])/4.])
                            S7set.extend([S7[int(float(x+i)/2.),int(float(y+j)/2.)]]) 
                            S8set.extend([S8[int(float(x+i)/2.),int(float(y+j)/2.)]])
                            S9set.extend([S9[int(float(x+i)/2.),int(float(y+j)/2.)]])
                                           
                    pixel_info.append([S1set, S2set, S3set, S4set, S5set, S6set,
                                      S7set, S8set, S9set, truth_set[int(float(y)/2.)+int(float(y)/2.)*(int(float(x)/2.-1.))]])                   
                                       
        else:
            pass
    #print(np.shape(pixel_info))
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
    
    os.chdir(olddir)
    return training_data, validation_data, training_truth, validation_truth


training_data, validation_data, training_truth, validation_truth = prep_data(directory)

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
