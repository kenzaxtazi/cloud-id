#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:48:30 2018

@author: kenzatazi
"""

# IDL code


import numpy as np


def vfm_feature_flags (val):
    """ Python version of the IDL code to read the bitwise flags"""
    
    print(val)
    
    feature_type = 0
    feature_type_qa = 0
    ice_water_phase = 0
    ice_water_phase_qa = 0
    feature_subtype = 0
    cloud_aerosol_psc_type_qa = 0
    horizontal_averaging = 0
    
    
    for i in range(0,16):
      if np.bitwise_and(val,2**i)!= 0:
        if i+1 == 1:
            feature_type = feature_type + 1
        if i+1 == 2:
            feature_type = feature_type + 2
        if i+1 == 3:
            feature_type = feature_type + 4
        if i+1 == 4:
            feature_type_qa = feature_type_qa + 1
        if i+1 == 5:
            feature_type_qa = feature_type_qa + 2
        if i+1 == 6:
            ice_water_phase = ice_water_phase + 1
        if i+1 == 7:
            ice_water_phase = ice_water_phase + 2
        if i+1 == 8:
            ice_water_phase_qa = ice_water_phase_qa + 1
        if i+1 == 9:
            ice_water_phase_qa = ice_water_phase_qa + 2
        if i+1 == 10:
            feature_subtype = feature_subtype + 1
        if i+1 == 11:
            feature_subtype = feature_subtype + 2
        if i+1 == 12:
            feature_subtype = feature_subtype + 4
        if i+1 == 13:
            cloud_aerosol_psc_type_qa = cloud_aerosol_psc_type_qa + 1
        if i+1 == 14:
            horizontal_averaging = horizontal_averaging + 1
        if i+1 == 15:
            horizontal_averaging = horizontal_averaging + 2
        if i+1 == 16:
            horizontal_averaging = horizontal_averaging + 4
            


    
    return feature_type

    
        
        
    
