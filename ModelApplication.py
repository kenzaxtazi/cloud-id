#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:54:58 2018

@author: kenzatazi
"""

import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import DataLoader as pim
#from cartopy import config
#import cartopy.crs as ccrs

# create readers and open files
#scn = Scene(filenames=glob('/home/hep/kt2015/cloud/SLSTR/2018/08/S3A_SL_1_RBT____20180823T041605_20180823T041905_20180824T083800_0179_035_033_1620_LN2_O_NT_003.SEN3/*'), 
#           reader='nc_slstr')

def apply_mask(model, scenes, bayesian=False, empirical=False):

    """
    Function to plot the model mask superimposed one or more SLSTR scenes and, 
    optionallly,  the associated baseyian and empircal cloud masks that come 
    with the file.
    
    Args:
        model: a tensorflow model object 
        scenes: a list of satpy Scene objects
    Kwargs:
        bayesian=False, if set to True the function will also plot the Bayesian
                        mask
        empirical=False,  if set to True the function will also plot the 
                        empirical mask
    Returns: 
        Plot of scenes with masks
    """    
    
    for scn in scenes: 

        scn.load(['S1_an','S2_an','S3_an','S4_an','S5_an','S6_an','S7_in','S8_in',
                  'S9_in','bayes_an', 'bayes_in','cloud_an', 'longitude_an', 
                  'latitude_an', 'confidence_an', 'satellite_zenith_angle', 
                  'solar_zenith_angle', ])
        
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
        confidence_an = np.nan_to_num(scn['confidence_an'].values)
        
        #mask= []
        inputs_list = [] 
        
        for i in tqdm(range(len(S1))):
            for j in range(len(S1[0])):
                inputs = np.array([S1[i,j],S2[i,j],S3[i,j],S4[i,j],S5[i,j],S6[i,j],
                                  S7[int(i/2.),int(j/2.)],S8[int(i/2.),int(j/2.)],
                                  S9[int(i/2.),int(j/2.)],
                                  salza[int(i/2.),int(j/2.)], solza[int(i/2.),int(j/2.)],
                                  lat[i,j],lon[i,j],
                                  confidence_an[i,j]])
                inputs_list.append(inputs)
        
        inputs_list = np.array(inputs_list)
        inputs_list = inputs_list.reshape(-1,1,14,1)
        
        labels =  model.predict_label(inputs_list) 
        
        mask = (np.array(labels[:,0])).reshape(2400,3000)
        #       label =  model.predict_label(inputs)
        #       mask.append(label)
        
        #mask = (np.array(mask)[:,:,0]).reshape(2400,3000)
        mask = np.ma.masked_where(mask < 1, mask) 


        plt.figure()
       
        '''
        # map 
        img_extent = (lon[0,0], lon[-1,-1], lat[0,0], lat[-1,-1])
        ax = plt.axes(projection=ccrs.PlateCarree())
    
        # set a margin around the data
        ax.set_xmargin(0.05)
        ax.set_ymargin(0.10)

        # add the image. Because this image was a tif, the "origin" of the image is in the
        # upper left corner
        #ax.imshow(S1, cmap='gray', origin='upper', extent=img_extent, transform=ccrs.PlateCarree())
        #ax.coastlines(resolution='500m', color='black', linewidth=1)
        '''
        plt.imshow(S1,cmap='gray')
        plt.imshow(mask, cmap='cool', vmin=1, vmax=2, alpha=0.3)
    
        if bayesian == True:
            bayes_mask= pim.create_mask(scn, 'bayes_in')
            bayes_mask = np.ma.masked_where(bayes_mask < 1, bayes_mask)
            mask(bayes_mask,'Baseyian mask', S1)
        
        if empirical  == True:
            emp_mask= pim.create_mask(scn, 'cloud_an')
            emp_mask = np.ma.masked_where(emp_mask < 1, emp_mask)
            pim.mask(emp_mask,'Empirical mask', S1)
        
        plt.show()
