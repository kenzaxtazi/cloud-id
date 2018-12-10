#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:54:58 2018

@author: kenzatazi
"""



# application

from satpy import Scene
from glob import glob
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm

# create readers and open files
scn = Scene(filenames=glob('/cloud/SLSTR/2018/08/S3A_SL_1_RBT____20180823T041605_20180823T041905_20180824T083800_0179_035_033_1620_LN2_O_NT_003.SEN3/*'), 
           reader='nc_slstr')

def apply_mask(model, scenes):
    
    for scn in scenes: 

        scn.load(['S1_an','S2_an','S3_an','S4_an','S5_an','S6_an','S7_in','S8_in',
                  'S9_in','bayes_an', 'bayes_in','cloud_an', 'longitude_an', 
                  'latitude_an', 'satellite_zenith_angle', 'solar_zenith_angle'])
        
        S1= np.nan_to_num(scn['S1_an'].values)
        S2= np.nan_to_num(scn['S2_an'].values)
        S3= np.nan_to_num(scn['S3_an'].values)
        S4= np.nan_to_num(scn['S4_an'].values)
        S5= np.nan_to_num(scn['S5_an'].values)
        S6= np.nan_to_num(scn['S6_an'].values)
        S7= np.nan_to_num(scn['S7_in'].values) 
        S8= np.nan_to_num(scn['S8_in'].values)
        S9= np.nan_to_num(scn['S9_in'].values)
        salza =  np.nan_to_num(scn['satellite_zenith_angle'].values)
        solza =  np.nan_to_num(scn['solar_zenith_angle'].values)
        lat =  np.nan_to_num(scn[ 'latitude_an'].values)
        lon =  np.nan_to_num(scn['longitude_an'].values)
        
        mask= []
        
        for i in tqdm(range(len(S1))):
            for j in range(len(S1[0])):
                inputs = np.array([S1[i,j],S2[i,j],S3[i,j],S4[i,j],S5[i,j],S6[i,j],
                                  S7[int(i/2.),int(j/2.)],S8[int(i/2.),int(j/2.)],
                                  S9[int(i/2.),int(j/2.)],salza[int(i/2.),int(j/2.)],
                                  solza[int(i/2.),int(j/2.)],
                                  lat[i,j],lon[i,j]])
                inputs = inputs.reshape(-1,1,13,1)
                label =  model.predict_label(inputs)
                mask.append(label)
        
        mask = (np.array(mask)[:,:,0]).reshape(2400,3000)
        
        plt.figure()
        plt.imshow(S1, 'gray')
        plt.imshow(mask, cmap='OrRd', alpha=0.3)
        plt.show()
