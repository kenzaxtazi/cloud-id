#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:50:15 2018

@author: kenzatazi
"""


from satpy import Scene
from glob import glob
import matplotlib.pyplot as plt
import numpy as np 

# create readers and open files
# scn = Scene(filenames=glob('/Users/kenzatazi/Downloads/S3A_SL_1_RBT____20180822T000619_20180822T000919_20180822T015223_0179_035_016_3240_SVL_O_NR_003.SEN3/*'), 
#            reader='nc_slstr')


# load datasets from input files
def load_scene(scn):
    """ Loads the information from the netcdf files in the folder"""
    #scn.load(scn.available_dataset_names())
    scn.load(['S1_an','S2_an','S3_an','S4_an','S5_an','S6_an','S7_in','S8_in',
              'S9_in','bayes_an', 'bayes_in','cloud_an', 'longitude_an', 
              'latitude_an', 'solar_zenith_angle'])
    

#load_scene(scn)

#S1= np.nan_to_num(scn['S1_an'].values)
#S2= np.nan_to_num(scn['S2_an'].values)
#S3= np.nan_to_num(scn['S3_an'].values)
#S4= np.nan_to_num(scn['S4_an'].values)
#S5= np.nan_to_num(scn['S5_an'].values)
#S6= np.nan_to_num(scn['S6_an'].values)
#S7= np.nan_to_num(np.array(scn['S7_in'][:-1])) 
#S8= np.nan_to_num(np.array(scn['S8_in'][:-1]))
#S9= np.nan_to_num(np.array(scn['S9_in'][:-1]))


def create_mask(scn, mask_name):
    """Extracts bitmasks and combines them into an overall mask array"""
    mask=[]
    for bitmask in scn[mask_name].flag_masks[:-2]:
        data = scn[mask_name].values & bitmask
        mask.append(data)
    mask= np.sum(mask, axis=0)
    return mask


#bayes_mask= create_mask(scn, 'bayes_in')
#emp_mask= create_mask(scn, 'cloud_an')


# single channel images 

# channel_arrays=[S1, S2, S3, S4, S5, S6, S7, S8, S9]

#for i in channel_arrays:
#    plt.figure()
#    plt.imshow(i, 'gray')


# false color image 

def norm(band):
    """ Normalises the bands for the false color image"""
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))


def false_color_image(band1, band2, band3, plot=True):
    """ 
    Creates a false colour image
    
    Input: 
    band1 (2D array) <=> red 
    band2 (2D array) <=> green
    band3 (2D array) <=> blue
    
    Output: 6D array (3*2D)
    
    if: plot= True, the image is plotted
    """
    rgb = np.dstack((norm(band1),norm(band2),norm(band3)))

    if plot == True:
        plt.figure()
        plt.imshow(rgb)
        plt.title('False colour image')
    
    return rgb


def mask(mask, mask_name, background):
    """Plots a semi-transparent mask over a background image"""
    plt.figure()
    plt.imshow(background, 'gray')
    plt.title(mask_name)
    plt.imshow(mask, vmax=1, cmap='OrRd', alpha=0.3)

        
#false_color_image(S3, S2, S1, plot=True)
#mask(bayes_mask,'Baseyian mask', S1)
#mask(emp_mask,'Empirical mask', S1)

plt.show()

