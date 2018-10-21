# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:46:43 2018

@author: tomzh
"""

import os
from satpy import Scene
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import platform
from time import time




def path_to_public():
    os.chdir()
    path = os.getcwd()
    if path[10] == 't': # Tom's directory
        path = path[:16] + "public_html"
    if path[10] == 'k': # Kenza's directory
        path = path[:17] + "public_html"
    return(path)
    
    
    
def scene_loader(path):
    # Returns a satpy scene object from the provided file
    Current_OS = platform.platform()
    if Current_OS[:6] == 'Darwin':
        print('Hi Kenza')
    if path[-1] == '/':
        path = path + "*"
    elif path[-1] == '*':
        pass
    else:
        path = path + "/*"
    filenames = glob(path)
    scn = Scene(filenames=filenames, reader='nc_slstr')
    return(scn)

def mask_analysis(scn):
    # Loads the a masks from simplistic and bayesian files. 
    # WIP: Creates figure of all simplistic masks
    scn.load(['cloud_an', 'bayes_an'])
    """ 
    Cloud_bn file
    Flag masks: Flag meanings
    1: Visible 1.37_threshold
    2: 1.6_small_histogram
    4: 1.6_large_histogram
    8: 2.25_small_histogram
    16: 2.25 large_histogram
    32: 11_spatial_coherence
    64: gross_cloud
    128: thin_cirrus
    256: medium_high
    512: fog_low_stratus
    1024: 11_12_view_difference
    2048: 3.7.11_view_difference
    4096: thermal_histogram
    16384: spare
    32768: spare
    
    Bayes_bn file
    Flag masks: Flag meanings
    1: single_low
    2: single_moderate
    4: dual_low
    8: dual_moderate
    16: spare
    32: spare
    64: spare
    128: spare
    """
    for mask in scn['cloud_bn'].flag_masks[:-2]:
        plt.figure()
        plt.title(str(mask))
        data = scn['cloud_bn'].values & mask
        plt.imshow(data)
        
    
    
    
def summary(scene, filenames=None, saveimage=False, outputpath='public'):
    # Loads positional S1_n channel data. Prints lat/lon of corner pixel
    # If saveimage is True, saves png to current directory with metadata
    scene.load(['S1_n', 'latitude', 'longitude'])
    lat = scene['latitude'].values[0][0] # Latitude of corner pixel
    lon = scene['longitude'].values[0][0] # Longitude of corner pixel
    if saveimage != False:
        if outputpath == 'public':
            # cd to public folder
            os.chdir(path_to_public())
        if filenames != None:
            imagename = ('S1n_' + str(filenames[0][:31]) + '_' + 
                         str(filenames[0][82:94]) + '-(' + str(lat) + ',' + 
                         str (lon) +')')
        else:
            imagename = 'test'
        scene.save_dataset('S1_n', str(imagename) + '.png')
    print(str(lat) + ', ' + str(lon))


def makepltimage(scene, channel='S1_n'):
    # Use matplotlib to produce image of specified channel
    data = scene[channel].values
    data = np.nan_to_num(data)
    plt.figure()
    plt.imshow(data)
    
    
def makepngimage(scene, channel='S1_n', outputpath='public'):
    if outputpath == 'public':
        # cd to public folder
        os.chdir(path_to_public())
    scene.save_dataset(channel, str(time()) + '.png')
    
    