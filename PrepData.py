#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:20:12 2018

@author: kenzatazi
"""

### Data prepareation

import numpy as np
import Plotting_images_and_masks as pim
import vfm_feature_flags as vfm 
import CalipsoReader2 as cr2
from pyhdf.SD import SD, SDC
from satpy import Scene
from random import shuffle
from tqdm import tqdm   #percentage bar for tasks.
import pandas as pd 


def open_matches():
    """ 
    Opens the matches file to get corresponding files 
    
    value0= CALIOP filename 
    value1= SLSTR filename
    value2= link to dowload SLSTR file 
    """
    array= np.loadtxt('/home/hep/trz15/Masters_Project/Matches.txt',dtype=str)
    filenames=[]
    for n in array:
        value0, value1, value2= n.split(',')
        filenames.append([value0, value1, value2])
    return filenames
    


def save_data(pixels):
    """
    Saves and return pixel values and metadata needed.
    
    Current format:
    
    array1: SLSTR 5x5km data in 9 channels (1*9*25)
    array2: feature classification flag from CALIOP (1*1)
    array3: datetimes for CALIOP and SLSTR respectively (1*2)
    
    """
    val= pixels[:,0]
    SLSTR_pathnames = pixels[:,1]
    CALIOP_pathnames = pixels[:,2]
    
    
    pixel_info=[]
    
    for f in range(len(val)):
        
        #Load SLSTR file 
        scn = Scene(filenames=SLSTR_pathnames[f], reader='nc_slstr')
        pim.load_scene(scn)
        slstr_datetime=(SLSTR_pathnames[f])[-83:-68]
        
        #Load CALIOP file 
        file = SD(CALIOP_pathnames[f], SDC.READ)
        data=cr2.load_data(file,'Feature_Classification_Flags')
        caliop_datetime= [] 

        
        y,x,v = val[f,0],val[f,1],val[f,2]  
        
        truth_set=[]        
        truth_set.append(vfm.vfm_feature_flags(data[v,0]))
        #in UTC time 
        caliop_datetime.append(cr2.load_data(file,'Profile_UTC_Time')[v,0])

        S1set=[]
        S2set=[]
        S3set=[]
        S4set=[]
        S5set=[]
        S6set=[]
        S7set=[]
        S8set=[]
        S9set=[]
                
        for i in range(-4,5,2):
            for j in range(-4,5,2):
                
                S1set.extend([float((scn['S1_an'])[x+i,y+j]+
                                    (scn['S1_an'])[x+i+1,y+j]+
                                    (scn['S1_an'])[x+i,y+1+j]+
                                    (scn['S1_an'])[x+i+1,y+j+1])/4.])
                S2set.extend([float((scn['S2_an'])[x+i,y+j]+
                                    (scn['S2_an'])[x+i+1,y+j]+
                                    (scn['S2_an'])[x+i,y+1+j]+
                                    (scn['S2_an'])[x+i+1,y+j+1])/4.])
                S3set.extend([float((scn['S3_an'])[x+i,y+j]+
                                    (scn['S3_an'])[x+i+1,y+i]+
                                    (scn['S3_an'])[x+i,y+1+j]+
                                    (scn['S3_an'])[x+i+1,y+j+1])/4.])
                S4set.extend([float((scn['S4_an'])[x+i,y+j]+
                                    (scn['S4_an'])[x+i+1,y+j]+
                                    (scn['S4_an'])[x+i,y+1+j]+
                                    (scn['S4_an'])[x+i+1,y+j+1])/4.])
                S5set.extend([float((scn['S5_an'])[x+i,y+j]+
                                    (scn['S5_an'])[x+i+1,y+j]+
                                    (scn['S5_an'])[x+i,y+1+j]+
                                    (scn['S5_an'])[x+i+1,y+i+1])/4.])
                S6set.extend([float((scn['S6_an'])[x+i,y+j]+
                                    (scn['S6_an'])[x+i+1,y+j]+
                                    (scn['S6_an'])[x+i,y+1+j]+
                                    (scn['S6_an'])[x+i+1,y+j+1])/4.])
                S7set.extend([(scn['S7_in'])[int(float(x+i)/2.),int(float(y+j)/2.)]]) 
                S8set.extend([(scn['S8_in'])[int(float(x+i)/2.),int(float(y+j)/2.)]])
                S9set.extend([(scn['S9_in'])[int(float(x+i)/2.),int(float(y+j)/2.)]])
                
        pixel_info.append([[S1set, S2set, S3set, S4set, S5set, S6set,
                            S7set, S8set, S9set], truth_set, [caliop_datetime, 
                                               slstr_datetime]]) 
    
    np.savetxt("/home/hep/trz15/Masters_Project/Collocated_pixels/pixel_info.csv", 
                      pixel_info, delimiter=",")
        
    return pixel_info
        



def prep_data(pixel_info):
    
    """ prepare data for one CALIOP and one SLSTR file at a time """
    
    
    shuffle(pixel_info)     # mix real good

    data= []
    truth= []
    
    for i in pixel_info:
        data.append(i[0])
        truth.append(i[1])
        
  
    training_data= np.array(data[:-500]) # take all but the 500 last 
    validation_data= np.array(data[-500:])    # take 500 last pixels 
    training_truth= np.array(truth[:-500])
    validation_truth= np.array(truth[-500:])
    
    return training_data, validation_data, training_truth, validation_truth