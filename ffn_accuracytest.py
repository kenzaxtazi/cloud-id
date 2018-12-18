#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:37:26 2018

@author: kenzatazi
"""

import pandas as pd
import numpy as np
from random import shuffle
import vfm_feature_flags2 as vfm 
import ModelEvaluation as me
import matplotlib.pyplot as plt
import sklearn.utils 
import ModelApplication as app
from satpy import Scene
from glob import glob


LR = 1e-3


scn1= Scene(filenames=glob('/home/hep/kt2015/cloud/SLSTR/2018/08/S3A_SL_1_RBT____20180823T041605_20180823T041905_20180824T083800_0179_035_033_1620_LN2_O_NT_003.SEN3/*'), 
           reader='nc_slstr')
scn2= Scene(filenames=glob('/home/hep/kt2015/cloud/SLSTR/2018/08/S3A_SL_1_RBT____20180829T200950_20180829T201250_20180831T004228_0179_035_128_1620_LN2_O_NT_003.SEN3/*'), 
           reader='nc_slstr')
scn3= Scene(filenames=glob('/home/hep/kt2015/cloud/SLSTR/2018/08/S3A_SL_1_RBT____20180829T200950_20180829T201250_20180831T004228_0179_035_128_1620_LN2_O_NT_003.SEN3/*'), 
           reader='nc_slstr')
scn4= Scene(filenames=glob('/home/hep/kt2015/cloud/SLSTR/2018/08/S3A_SL_1_RBT____20180829T200950_20180829T201250_20180831T004228_0179_035_128_1620_LN2_O_NT_003.SEN3/*'), 
           reader='nc_slstr')
scn5= Scene(filenames=glob('/home/hep/kt2015/cloud/SLSTR/2018/08/S3A_SL_1_RBT____20180829T200950_20180829T201250_20180831T004228_0179_035_128_1620_LN2_O_NT_003.SEN3/*'), 
           reader='nc_slstr')

scenes= [scn1, scn2, scn3, scn4, scn5]
 
MODEL_NAME = 'ffn_withancillarydata'.format(LR, 'feedforward') 

pixel_info1 = pd.read_pickle("/home/hep/trz15/Matched_Pixels/AprP1.pkl")
pixel_info2 = pd.read_pickle("/home/hep/trz15/Matched_Pixels/Run2P1.pkl")
pixel_info3 = pd.read_pickle("/home/hep/trz15/Matched_Pixels/FebP1.pkl") 
pixel_info4 = pd.read_pickle("/home/hep/trz15/Matched_Pixels/MarP1.pkl")
pixel_info5 = pd.read_pickle("/home/hep/trz15/Matched_Pixels/JunP1.pkl")
pixel_info6 = pd.read_pickle("/home/hep/trz15/Matched_Pixels/JulP1.pkl") 
pixel_info7 = pd.read_pickle("/home/hep/trz15/Matched_Pixels/AugP1.pkl") 
pixel_info8 = pd.read_pickle("/home/hep/trz15/Matched_Pixels/JanP1.pkl")  

pixel_info = pd.concat([pixel_info1,pixel_info2,pixel_info3,pixel_info4,
                        pixel_info5,pixel_info6,pixel_info7,pixel_info8])

pixels = sklearn.utils.shuffle(pixel_info[:-100])

pixel_values = (pixels[['S1_an','S2_an','S3_an','S4_an','S5_an','S6_an',
                            'S7_in','S8_in','S9_in',
                            'satellite_zenith_angle', 'solar_zenith_angle', 
                            'latitude_an', 'longitude_an',
                            'Feature_Classification_Flags',
                            'TimeDiff']]).values

time_slices = np.linspace(0,3600,10)
accuracies = []



for k in range(10):
    
    for t in time_slices:
        
        # slices 
        p = [pix for pix in pixel_values if [t<=abs(pix[:,10])<t+360] 
        
        def prep_data(pixel_info):
            
            """ 
            Prepares data for matched SLSTR and CALIOP pixels into training data, 
            validation data, training truth data, validation truth data.
            
            """
            
            shuffle(pixel_info)     # mix real good
            
            conv_pixels= pixel_info.astype(float)
            pix= np.nan_to_num(conv_pixels)
            
            data = pix[:,:9]
            truth_flags = pix[:,9]
            
            truth_oh=[]
            
            for d in truth_flags:
                i = vfm.vfm_feature_flags(int(d))
                if i == 2:
                    truth_oh.append([1.,0.])    # cloud 
                if i != 2:
                    truth_oh.append([0.,1.])    # not cloud 
                
          
            pct = int(len(data)*.15)
            training_data= np.array(data[:-pct])    # take all but the last 15%  
            validation_data= np.array(data[-pct:])    # take the last 15%
            training_truth= np.array(truth_oh[:-pct])
            validation_truth= np.array(truth_oh[-pct:])
            
            return training_data, validation_data, training_truth, validation_truth
                
        
        
        # prepares data for cnn 
        
        training_data, validation_data, training_truth, validation_truth = prep_data(p)
        
        
        
        #### MACHINE LEARNING 
        
        import tflearn 
        from tensorflow import reset_default_graph
        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.estimator import regression
        
        
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
        
        
        network = regression(softmax, optimizer='Adam', learning_rate=LR,
                             loss='categorical_crossentropy', name='targets')
        
        model = tflearn.DNN(network, tensorboard_verbose=0)
        
        
        
        ### UNPACK SAVED DATA
        
        
        model.fit(training_data, training_truth, n_epoch=2, validation_set =
                  (validation_data, validation_truth), snapshot_step=1000, 
                  show_metric=False, run_id=MODEL_NAME)
        
        acc= me.get_accuracy(model,validation_data,validation_truth)        
        accuracies.append(acc)

        app.apply_mask(model, scenes, bayesian=True)

        reset_default_graph()
    
    plt.figure('Time difference vs accuracy')
    plt.title('Time difference as a function of accuracy')
    plt.xlabel('Time difference (s)')
    plt.ylabel('Accuracy')
    plt.plot(time_slices,accuracies)

    
        


    
    