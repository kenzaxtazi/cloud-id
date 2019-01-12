#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:54:58 2018

@author: kenzatazi
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from DataLoader import scene_loader, upscale_repeat
import matplotlib.animation as animation


def apply_mask(model, Sfile, model_inputs=13, plot=True, bayesian=False, empirical=False):
    """
    Function to produce predicted mask for given model and SLSTR file. 

    Produces plot of the output mask overlayed on S1 channel data by default.
    Can also produce plots of SLSTR's included masks.

    Parameters
    ----------
    model: tflearn.DNN model object
        A trained tflearn model object which produces masks for N pixels given an (N, 1, model_inputs, 1) shaped tensor as input.
        Such models are produced by ffn.py and can be loaded from a local file using ModelApplication.py

    Sfile: str
        A path to an SLSTR file folder. 

    model_inputs: int, optional
        Number of inputs model was trained on.
        Default is 13

    plot: bool, optional
        If True, produce plot of output mask overlayed on S1 channel. Also enable plotting of bayesian and empirical masks.
        Default is True

    bayesian: bool, optional
        If True, produce plot of SLSTR Bayesian mask
        Default is False

    empirical: bool, optional
        If True, produce plot of SLSTR empirical mask
        Default is False

    Returns
    -------
    mask: array
        Mask predicted by model for Sfile
    """
    scn = scene_loader(Sfile)

    scn.load(['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in', 'bayes_an',
              'bayes_in', 'cloud_an', 'latitude_an', 'longitude_an', 'satellite_zenith_angle', 'solar_zenith_angle', 'confidence_an'])

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
    confidence = np.nan_to_num(scn['confidence_an'].values)

    if model_inputs == 13:
        inputs = np.array([S1, S2, S3, S4, S5, S6, S7,
                           S8, S9, salza, solza, lat, lon])
        inputs = np.swapaxes(inputs, 0, 2)
        inputs = inputs.reshape((-1, 1, 13, 1), order='F')

    elif model_inputs == 14:
        inputs = np.array([S1, S2, S3, S4, S5, S6, S7,
                           S8, S9, salza, solza, lat, lon, confidence])
        inputs = np.swapaxes(inputs, 0, 2)
        inputs = inputs.reshape((-1, 1, 14, 1), order='F')

    label = model.predict_label(inputs)

    mask = np.array(label)
    mask = mask[:, 0].reshape(2400, 3000)

    # bayes_mask = []
    # for bitmask in scn['bayes_in'].flag_masks[:-2]:
    #     data = scn['bayes_in'].values & bitmask
    #     bayes_mask.append(data)
    # bayes_mask = np.sum(bayes_mask, axis=0)
    #    plt.imshow(bayes_mask, cmap='OrRd', alpha=0.3)

    if plot is True:
        fig = plt.figure()

        # plt.figure()
        # plt.imshow(S1, 'gray')
        # plt.imshow(mask, vmax=1, cmap='Blues', alpha=0.3)
        # plt.title('Composite')

        # plt.figure()
        im1 = [plt.imshow(S1, 'gray')]
        # plt.title('S1_an channel data')

        # plt.figure()
        im2 = [plt.imshow(mask, cmap='Blues')]
        # plt.title('Cloud mask output')
        # plt.show()

        # ims = [im1, im2]
        # ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=0)

        bmask = upscale_repeat(np.nan_to_num(scn['bayes_in'].values))
        bmask = bmask.astype(int)
        bmask = bmask & 2
        bmask = bmask / 2
        bmask = np.ones(bmask.shape) - bmask
        im3 = [plt.imshow(bmask, cmap='Reds')]
        ims = [im1, im2, im1, im3]
        ani = animation.ArtistAnimation(
            fig, ims, interval=500, blit=True, repeat_delay=0)
        plt.show()

    return(mask)
