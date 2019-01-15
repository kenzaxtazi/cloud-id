#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:54:58 2018

@author: kenzatazi
"""

import numpy as np
from DataLoader import scene_loader, upscale_repeat
import DataPreparation as dp


def apply_mask(model, Sfile, num_inputs=13):
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
    if num_inputs == 13:
        inputs = dp.get13inputs(Sfile)
    elif num_inputs == 14:
        inputs = dp.get14inputs(Sfile)

    label = model.predict_label(inputs)

    mask = np.array(label)
    mask = mask[:, 0].reshape(2400, 3000)

    return(mask)
