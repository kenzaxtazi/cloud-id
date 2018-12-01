#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:48:30 2018

@author: kenzatazi
"""

# IDL code


import numpy as np


def vfm_feature_flags(val):
    """ Python version of the IDL code to read the bitwise flags"""

    feature_type = val & 7
    feature_type_qa = (val >> 3) & 3
    ice_water_phase = (val >> 5) & 3
    ice_water_phase_qa = (val >> 7) & 3
    feature_subtype = (val >> 9) & 7
    cloud_aerosol_psc_type_qa = (val >> 12) & 1
    horizontal_averaging = (val >> 13) & 7

    # if feature_type == 0:
    #     print("Feature Type : invalid (bad or missing data)")
        
    # if feature_type == 1:
    #     print("Feature Type : clean air")
        
    # if feature_type == 2:
    #     print("Feature Type : cloud")
    #     if feature_subtype == 0:
    #         print("Feature Subtype : low overcast, transparent")
    #     if feature_subtype == 1:
    #         print("Feature Subtype : low overcast, opaque")
    #     if feature_subtype == 2:
    #         print("Feature Subtype : transition stratocumulus")
    #     if feature_subtype == 3:  
    #         print("Feature Subtype : low, broken cumulus")
    #     if feature_subtype == 4:
    #         print("Feature Subtype : altocumulus (transparent)")
    #     if feature_subtype == 5:
    #         print("Feature Subtype : altostratus (opaque)")
    #     if feature_subtype == 6:
    #         print("Feature Subtype : cirrus (transparent)")
    #     if feature_subtype == 7:
    #         print("Feature Subtype : deep convective (opaque)")
    #     else : 
    #         print("*** error getting Feature Subtype")
            
    # if feature_type == 3:
    #     print("Feature Type : aerosol")
    #     if feature_subtype == 0:
    #         print("Feature Subtype : not determined")
    #     if feature_subtype == 1:  
    #         print("Feature Subtype : clean marine")
    #     if feature_subtype == 2: 
    #         print("Feature Subtype : dust")
    #     if feature_subtype == 3: 
    #         print("Feature Subtype : polluted continental")
    #     if feature_subtype == 4: 
    #         print("Feature Subtype : clean continental")
    #     if feature_subtype == 5: 
    #         print("Feature Subtype : polluted dust")
    #     if feature_subtype == 6:
    #         print("Feature Subtype : smoke")
    #     if feature_subtype == 7: 
    #         print("Feature Subtype : other")
    #     else : 
    #         print("*** error getting Feature Subtype")
            
    # if feature_type == 4:
    #     print("Feature Type : stratospheric feature--PSC or stratospheric aerosol")
    #     if feature_subtype == 0:
    #         print("Feature Subtype : not determined")
    #     if feature_subtype == 1:
    #         print("Feature Subtype : non-depolarizing PSC")
    #     if feature_subtype == 2:
    #         print ("Feature Subtype : depolarizing PSC")
    #     if feature_subtype == 3:
    #         print, "Feature Subtype : non-depolarizing aerosol"
    #     if feature_subtype == 4:
    #         print("Feature Subtype : depolarizing aerosol")
    #     if feature_subtype == 5:
    #         print("Feature Subtype : spare")
    #     if feature_subtype == 6:
    #         print("Feature Subtype : spare")
    #     if feature_subtype == 7:
    #         print("Feature Subtype : other")
    #     else: 
    #         print("*** error getting Feature Subtype")
    
    # if feature_type == 5:
    #     print("Feature Type : surface")
    # if feature_type == 6:
    #     print("Feature Type : subsurface")
    # if feature_type == 7:
    #     print("Feature Type : no signal (totally attenuated)")
    # else : 
    #     print("*** error getting Feature Type")
        
    
    # if feature_type_qa == 0:
    #     print("Feature Type QA : none")
    # if feature_type_qa == 1:
    #     print("Feature Type QA : low")
    # if feature_type_qa == 2:
    #     print("Feature Type QA : medium")
    # if feature_type_qa == 3:
    #     print("Feature Type QA : high")
    # else : 
    #     print("*** error getting Feature Type QA")
        
    # if ice_water_phase == 0:
    #     print("Ice/Water Phase : unknown/not determined")
    # if ice_water_phase == 1:
    #     print("Ice/Water Phase : ice")
    # if ice_water_phase == 2:
    #     print("Ice/Water Phase : water")
    # if ice_water_phase == 3:
    #     print("Ice/Water Phase : mixed phase")
    # else : 
    #     print("*** error getting Ice/Water Phase")

    # if ice_water_phase_qa == 0:
    #     print("Ice/Water Phase QA: none")
    # if ice_water_phase_qa == 1:
    #     print("Ice/Water Phase QA: low")
    # if ice_water_phase_qa == 2:
    #     print("Ice/Water Phase QA: medium")
    # if ice_water_phase_qa == 3:
    #     print("Ice/Water Phase QA: high")
    # else : 
    #     print("*** error getting Ice/Water Phase QA")


    # if cloud_aerosol_psc_type_qa == 0:
    #     print("Cloud/Aerosol/PSC Type QA : not confident")
    # else:
    #     print("Cloud/Aerosol/PSC Type QA : confident")

    # if horizontal_averaging == 0:
    #     print("Horizontal averaging required for detection: not applicable")
    # if horizontal_averaging == 1: 
    #     print("Horizontal averaging required for detection: 1/3 km")
    # if horizontal_averaging == 2:
    #     print("Horizontal averaging required for detection: 1 km")
    # if horizontal_averaging == 3:
    #     print("Horizontal averaging required for detection: 5 km")
    # if horizontal_averaging == 4:
    #     print("Horizontal averaging required for detection: 20 km")
    # if horizontal_averaging == 5:
    #     print("Horizontal averaging required for detection: 80 km")
    # else : 
    #     print("*** error getting Horizontal averaging")
    return(feature_type)
