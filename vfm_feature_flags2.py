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

    return(feature_type)
