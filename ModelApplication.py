#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:54:58 2018

@author: kenzatazi
"""

import numpy as np
import DataPreparation as dp


def apply_mask(model, Sfile, input_type=24, modeltype='ffn', positions=None):
    """
    Function to produce predicted mask for given model and SLSTR file.

    Produces plot of the output mask overlayed on S1 channel data by default.
    Can also produce plots of SLSTR's included masks.

    Parameters
    ----------
    model: tflearn.DNN model object
        A trained tflearn model object which produces masks for N pixels given
        an (N, 1, model_inputs, 1) shaped tensor as input. Such models are
        produced by ffn.py and can be loaded from a local file using
        ModelApplication.py

    Sfile: str
        A path to an SLSTR file folder.

    Returns
    -------
    mask: array
        Mask predicted by model for Sfile
    """
    if modeltype == 'ffn':
        inputs = dp.getinputs(Sfile, input_type)
    if modeltype == 'cnn':
        inputs = dp.cnn_getinputs(Sfile, positions)
    
    returnlist = []

    label = model.predict_label(inputs)
    lmask = np.array(label)
    lmask = lmask[:, 0].reshape(2400, 3000)
    returnlist.append(lmask)

    prob = model.predict(inputs)
    pmask = np.array(prob)
    pmask = pmask[:, 0].reshape(2400, 3000)
    returnlist.append(pmask)

    return(returnlist)
