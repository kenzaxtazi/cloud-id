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
from satpy import DatasetID

# create readers and open files
scn = Scene(filenames=glob('/Users/kenzatazi/Downloads/S3A_SL_1_RBT____20180822T000619_20180822T000919_20180822T015223_0179_035_016_3240_SVL_O_NR_003.SEN3/*'), 
            reader='nc_slstr')


# load datasets from input files
scn.load(['S1_n','S2_n','S3_n','S4_bn','S5_bn','S6_bn','S7','S8','S9',
          'longitude', 'latitude', 'bayes_bn','cloud_bn'])

#print(scn['S1_bn'])

S1= np.nan_to_num(scn['S1_n'].values)
S2= np.nan_to_num(scn['S2_n'].values)
S3= np.nan_to_num(scn['S3_n'].values)
S4= np.nan_to_num(scn['S4_bn'].values)
S5= np.nan_to_num(scn['S5_bn'].values)
S6= np.nan_to_num(scn['S6_bn'].values)
bayes_mask= np.nan_to_num(scn['bayes_bn'].values)
emp_mask= np.nan_to_num(scn['cloud_bn'].values)

# cannot access channels 7-9 

#S7= np.nan_to_num(np.array(scn['S7'][:-1])) 
#S8= np.nan_to_num(np.array(scn['S8'][:-1]))
#S9= np.nan_to_num(np.array(scn['S9'][:-1]))


# single channel images 

channel_arrays=[S1, S2, S3, S4, S5, S6] #S7, S8, S9]

for i in channel_arrays:
    plt.figure()
    plt.imshow(i, 'gray')


# false color image 

def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))


rgb = np.dstack((norm(S3),norm(S2),norm(S1)))
  
plt.figure()
plt.imshow(rgb)



# complete bayesian cloud mask
plt.figure('Complete bayesian cloud mask')
transparency_values= np.ones([2400,3000])*0.1
plt.imshow(S1,'gray')
plt.imshow(bayes_mask, alpha=0.3)

# complete empirical cloud mask
plt.figure('Complete empirical cloud mask')
plt.imshow(S1, 'gray')

plt.imshow(emp_mask, alpha=0.3)


# bit of baseyian cloud mask 


plt.show()
