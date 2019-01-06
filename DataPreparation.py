#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:29:23 2019

@author: kenzatazi
"""

import numpy as np
import vfm_feature_flags2 as vfm 

def prep_data(pixel_info):
    
    """ 
    Prepares data for matched SLSTR and CALIOP pixels into training data, 
    validation data, training truth data, validation truth data.
    
    """
    
    # shuffle(pixel_info)     # mix real good
    
    conv_pixels= pixel_info.astype(float)
    pix= np.nan_to_num(conv_pixels)
    
    data = pix[:,:-2]
    truth_flags = pix[:,-2]
    
    truth_oh=[]

    for d in truth_flags:
        i = vfm.vfm_feature_flags(int(d))
        if i == 2:
            truth_oh.append([1.,0.])    # cloud 
        if i != 2:
            truth_oh.append([0.,1.])    # not cloud 
    
    pct = int(len(data)*.15)
    training_data= np.array(data[:-pct])    # take all but the 15% last 
    validation_data= np.array(data[-pct:])    # take the last 15% of pixels 
    training_truth= np.array(truth_oh[:-pct])
    validation_truth= np.array(truth_oh[-pct:])
    
    return training_data, validation_data, training_truth, validation_truth

def surftype_processing(dataframe):
    """
    Bitwise processing of SLSTR surface data. 
    
    Input: Dataframe of matched pixel information 
    Output: Dataframe of matched pixel information 
    """
    
    