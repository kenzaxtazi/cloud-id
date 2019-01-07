#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:54:58 2018

@author: kenzatazi
"""

from satpy import Scene
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# create readers and open files
# scn = Scene(filenames=glob('/home/hep/kt2015/cloud/SLSTR/2018/08/S3A_SL_1_RBT____20180823T041605_20180823T041905_20180824T083800_0179_035_033_1620_LN2_O_NT_003.SEN3/*'),
#           reader='nc_slstr')


def upscale_repeat(x, h=2, w=2):
    """Upscales an array, credit to https://stackoverflow.com/questions/46215414/upscaling-a-numpy-array-and-evenly-distributing-values"""
    return(x.repeat(h, axis=0).repeat(w, axis=1))


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

        scn.load(['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in',
                  'S9_in', 'bayes_an', 'bayes_in', 'cloud_an', 'longitude_an',
                  'latitude_an', 'satellite_zenith_angle', 'solar_zenith_angle'])

        S1 = np.nan_to_num(scn['S1_an'].values)
        S2 = np.nan_to_num(scn['S2_an'].values)
        S3 = np.nan_to_num(scn['S3_an'].values)
        S4 = np.nan_to_num(scn['S4_an'].values)
        S5 = np.nan_to_num(scn['S5_an'].values)
        S6 = np.nan_to_num(scn['S6_an'].values)
        S7 = upscale_repeat(np.nan_to_num(scn['S7_in'].values))
        S8 = upscale_repeat(np.nan_to_num(scn['S8_in'].values))
        S9 = upscale_repeat(np.nan_to_num(scn['S9_in'].values))
        salza = upscale_repeat(np.nan_to_num(
            scn['satellite_zenith_angle'].values))
        solza = upscale_repeat(np.nan_to_num(scn['solar_zenith_angle'].values))
        lat = np.nan_to_num(scn['latitude_an'].values)
        lon = np.nan_to_num(scn['longitude_an'].values)

        mask = []

        inputs = np.array([S1, S2, S3, S4, S5, S6,
                           S7, S8,
                           S9, salza,
                           solza,
                           lat, lon])
        inputs = np.swapaxes(inputs, 0, 1)
        inputs = np.swapaxes(inputs, 1, 2)
        inputs = inputs.reshape(-1, 1, 13, 1)
        print(inputs.shape)
        label = model.predict_label(inputs)
        mask.extend(label)

        mask = np.array(mask)
        mask = mask[:, 0].reshape(2400, 3000)

        bayes_mask = []
        for bitmask in scn['bayes_in'].flag_masks[:-2]:
            data = scn['bayes_in'].values & bitmask
            bayes_mask.append(data)
        bayes_mask = np.sum(bayes_mask, axis=0)

        plt.figure()
        plt.imshow(S1, 'gray')
        plt.imshow(mask, vmax=1, cmap='Blues', alpha=0.3)
        plt.imshow(bayes_mask, cmap='OrRd', alpha=0.3)
    plt.show()
