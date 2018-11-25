#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:20:12 2018

@author: kenzatazi
"""

### Data prepareation

import gc
import numpy as np
import Plotting_images_and_masks as pim
import vfm_feature_flags2 as vfm 
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
    df= pd.read_csv('/home/hep/trz15/Masters_Project/Matches.txt',dtype=str)
    array= df.values
    filenames=[]
    for n in array:
        filenames.append([n[0],n[1],n[2]])
    return filenames
    


def save_data(pixels):
    """
    Saves and return pixel values and metadata needed.
    
    Current format:
    
    array1: SLSTR 5x5km data in 9 channels (1*9*25)
    array2: feature classification flag from CALIOP (1*1)
    array3: datetimes for SLSTR and CALIOP respectively (1*2)
    array4: coordinates for SLSTR and CALIOP respectively (lat, lon) (1*2*2)
    array5: solar zenith angle for SLSTR and CALIOP respectively (1*2)
    
    """
    #write file
    df = pd.Dataframe([])
    df.to_csv("/home/hep/trz15/Collocated_Pixels/pixel_info_k.csv", mode='w', 
              delimiter=",")
    
    val= pixels[:,0]
    SLSTR_pathnames = pixels[:,1]
    CALIOP_pathnames = pixels[:,2]
        
    for f in tqdm(range(len(val))):
        
        #Load SLSTR file 
        scn = Scene(filenames=SLSTR_pathnames[f], reader='nc_slstr')
        pim.load_scene(scn)
        slstr_datetime=(SLSTR_pathnames[f])[-83:-68]
        
        #Load CALIOP file
        with cr2.SDopener(CALIOP_pathnames) as file:  
            flags = cr2.load_data(file,'Feature_Classification_Flags')
            longitudes = cr2.load_data(file,'Longitude')
            latitudes = cr2.load_data(file,'Latitudes')
            times = cr2.load_data(file,'Profile_UTC_Time')
            szas = cr2.load_data(file,'Solar_Zenith_Angle')
        
        # Saving matched pixel information 
        
        x,y,v = val[f,0],val[f,1],val[f,2]  
        
        truth_set= vfm.vfm_feature_flags(flags[v,0]) #in UTC time 
        caliop_datetime = times[v,0]
        caliop_lon = longitudes[v,0]
        caliop_lat = latitudes[v,0]
        caliop_sza = szas[v,0]
        
        slstr_lat= (scn['latitude_an'])[x,y]
        slstr_lon= (scn['longitude_an'])[x,y]
        slstr_sza= (scn['solar_zenith_angle'])[x,y]
        
        S1= np.nan_to_num(scn['S1_an'].values)
        S2= np.nan_to_num(scn['S2_an'].values)
        S3= np.nan_to_num(scn['S3_an'].values)
        S4= np.nan_to_num(scn['S4_an'].values)
        S5= np.nan_to_num(scn['S5_an'].values)
        S6= np.nan_to_num(scn['S6_an'].values)
        S7= np.nan_to_num(scn['S7_in'].values) 
        S8= np.nan_to_num(scn['S8_in'].values)
        S9= np.nan_to_num(scn['S9_in'].values)
                
        S1set=[]
        S2set=[]
        S3set=[]
        S4set=[]
        S5set=[]
        S6set=[]
        S7set=[]
        S8set=[]
        S9set=[]
        
        new_x= x-5
        new_y= y-5
        
        for i in range(0,10,1):
            for j in range(0,10,1):
                
                S1set.extend([float(S1[x+i,y+j])])
                S2set.extend([float(S2[x+i,y+j])])
                S3set.extend([float(S3[x+i,y+j])])
                S4set.extend([float(S4[x+i,y+j])])
                S5set.extend([float(S5[x+i,y+j])])
                S6set.extend([float(S6[x+i,y+j])])
                S7set.extend([S7[int(float(x+i)/2.),int(float(y+j)/2.)]]) 
                S8set.extend([S8[int(float(x+i)/2.),int(float(y+j)/2.)]])
                S9set.extend([S9[int(float(x+i)/2.),int(float(y+j)/2.)]])
                
        pixel_info = [[S1set, S2set, S3set, S4set, S5set, S6set, S7set, S8set, S9set], 
                      truth_set, 
                      [slstr_datetime, caliop_datetime],
                      [[slstr_lat, slstr_lon], [caliop_lat, caliop_lon]]
                      [slstr_sza, caliop_sza]]
        
        scn.unload()
        
        df= pd.DataFrame(pixel_info)
        df.to_csv("/home/hep/trz15/Collocated_Pixels/pixel_info_k.csv", mode='a'
                 delimiter=",")
        
    print('done')
        



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