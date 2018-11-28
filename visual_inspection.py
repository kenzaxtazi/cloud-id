#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:54:45 2018

@author: kenzatazi
"""


import Collocation2 as c
import Plotting_images_and_masks as pim
import CalipsoReader2 as cr2
import vfm_feature_flags2 as vfm 
from glob import glob
import numpy as np
from satpy import Scene
import matplotlib.pyplot as plt


def vis_inspection(model, test_set):
           
     #get SLTSR and caliop file 
     
     slstr = (test_set['Sfilename'].values)[0]
     caliop = (test_set['Cfilename'].values)[0]
     
     caliop_directory= '/home/hep/trz15/cloud/Calipso/1km/2018/04'
     slstr_directory= '/home/hep/trz15/cloud/SLSTR/2018/04'
     
     CALIOP_pathname= caliop_directory+'/'+ caliop[43:45]+'/'+caliop
     SLSTR_pathname= slstr_directory +'/'+ slstr+ '/*'
     
     pixels = c.collocate(SLSTR_pathname, CALIOP_pathname)
     
     
     scn = Scene(filenames=glob(SLSTR_pathname), reader='nc_slstr')
     pim.load_scene(scn)
     
     
     S1= np.nan_to_num(scn['S1_an'].values)
     S2= np.nan_to_num(scn['S2_an'].values)
     S3= np.nan_to_num(scn['S3_an'].values)
     S4= np.nan_to_num(scn['S4_an'].values)
     S5= np.nan_to_num(scn['S5_an'].values)
     S6= np.nan_to_num(scn['S6_an'].values)
     S7= np.nan_to_num(scn['S7_in'].values) 
     S8= np.nan_to_num(scn['S8_in'].values)
     S9= np.nan_to_num(scn['S9_in'].values)
     
     plt.figure()
     plt.imshow(S1, 'gray')
     
     with cr2.SDopener(CALIOP_pathname) as file:  
         flags = cr2.load_data(file,'Feature_Classification_Flags')
     
     model_input=[]
     truth_input=[]
     
     for val in pixels:
         model_input.append([S1[val[0],val[1]], S2[val[0],val[1]],
                             S3[val[0],val[1]], S4[val[0],val[1]],
                             S5[val[0],val[1]], S6[val[0],val[1]],
                             S7[int(float(val[0])/2.),int(float(val[1])/2.)],
                             S8[int(float(val[0])/2.),int(float(val[1])/2.)],
                             S9[int(float(val[0])/2.),int(float(val[1])/2.)]])
        
         i= vfm.vfm_feature_flags(flags[val[2],0])
        
         if i == 2:
             truth_input=[1.,0.]    # cloud 
        
         if i != 2:
             truth_input=[0.,1.]    # not cloud 
            
     model_input= (np.array(model_input)).reshape(-1,1,9,1)    
     predictions= model.predict_label(model_input)
     dots= np.array(truth_input) + np.array(predictions)
    
    
     for i in range(len(dots)):
        if dots[i,0] == 2.:
            plt.scatter(pixels[i,0],pixels[i,1], c='lightgreen',
                        edgecolors='green')
        if dots[i,0] == 0.:   
            plt.scatter(pixels[i,0],pixels[i,1], c='pink',
                        edgecolors='red')
        
        if dots[i,0] == 1.:  
            if truth_input[0] == 1.:
                plt.scatter(pixels[i,0],pixels[i,1], c='lightgreen',
                        edgecolors='red')   
            if truth_input[0] == 0.:
                plt.scatter(pixels[i,0],pixels[i,1], c='pink',
                        edgecolors='green')
                
         
        
     plt.show()
     plt.savefig('/home/hep/kt2015/cloud/ploooot.png')
       
                       
                       
                       
                
     
    