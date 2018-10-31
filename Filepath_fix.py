# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:20:51 2018

@author: tomzh
"""
import os
from satpy import Scene
from glob import glob
import matplotlib.pyplot as plt
import numpy as np



def fixdir(list_in):
    for i in range(len(list_in)):
        list_in[i] = list_in[i].replace('\\', '/')
    return(list_in)
    
def summary(scene, filenames=None, saveimage=False):
    scene.load(['S1_n', 'latitude', 'longitude'])
    lat = scene['latitude'].values[0][0] # Latitude of corner pixel
    lon = scene['longitude'].values[0][0] # Longitude of corner pixel
    if saveimage != False:
        if filenames != None:
            imagename = ('S1n_' + str(filenames[0][:31]) + '_' + 
                         str(filenames[0][82:94]) + '-(' + str(lat) + ',' + 
                         str (lon) +')')
        else:
            imagename = 'test'
        scene.save_dataset('S1_n', str(imagename) + '.png')
    print(str(lat) + ', ' + str(lon))

def mask_analysis(scene):
    scn.load(['cloud_bn', 'bayes_bn'])
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
    
    
    
def makeimage(scene, channel='S1_n'):
    data = scene[channel].values
    data = np.nan_to_num(data)
    plt.figure()
    plt.imshow(data)



    
slstr_channels = ['S1_n', 'S1_o', 'S2_n', 'S2_o', 'S3_n', 'S3_o', 'S4_an',
 'S4_ao', 'S4_bn', 'S4_bo', 'S4_cn', 'S4_co', 'S5_an', 'S5_ao', 'S5_bn',
 'S5_bo', 'S5_cn', 'S5_co', 'S6_an', 'S6_ao', 'S6_bn', 'S6_bo', 'S6_cn',
 'S6_co', 'S7', 'S8', 'S9']


#os.chdir("D:/SLSTR")
filenames = glob(r"D:\SLSTR\S3A_SL_1_RBT____20181023T210946_20181023T211246_20181023T234012_0179_037_143_0720_SVL_O_NR_003.SEN3\*")
filenames = fixdir(filenames)
scn = Scene(filenames=filenames, reader='nc_slstr')

