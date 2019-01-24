#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:29:23 2019

@author: kenzatazi
"""

import numpy as np
import DataLoader as DL
import datetime


def getinputs(Sreference, num_inputs=13):
    """
    For a given SLSTR file, produce a correctly formatted input array for
    tflearn model
    """
    if type(Sreference) == str:
        scn = DL.scene_loader(Sreference)
    else:
        scn = Sreference

    scn.load(['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in',
              'S8_in', 'S9_in', 'bayes_an', 'bayes_in', 'cloud_an',
              'latitude_an', 'longitude_an', 'satellite_zenith_angle',
              'solar_zenith_angle'])

    S1 = np.nan_to_num(scn['S1_an'].values)
    S2 = np.nan_to_num(scn['S2_an'].values)
    S3 = np.nan_to_num(scn['S3_an'].values)
    S4 = np.nan_to_num(scn['S4_an'].values)
    S5 = np.nan_to_num(scn['S5_an'].values)
    S6 = np.nan_to_num(scn['S6_an'].values)
    S7 = DL.upscale_repeat(np.nan_to_num(scn['S7_in'].values))
    S8 = DL.upscale_repeat(np.nan_to_num(scn['S8_in'].values))
    S9 = DL.upscale_repeat(np.nan_to_num(scn['S9_in'].values))
    salza = DL.upscale_repeat(np.nan_to_num(
        scn['satellite_zenith_angle'].values))
    solza = DL.upscale_repeat(np.nan_to_num(scn['solar_zenith_angle'].values))
    lat = np.nan_to_num(scn['latitude_an'].values)
    lon = np.nan_to_num(scn['longitude_an'].values)

    if num_inputs == 13:
        inputs = np.array([S1, S2, S3, S4, S5, S6, S7, S8, S9, salza,
                           solza, lat, lon])
        inputs = np.swapaxes(inputs, 0, 2)
        inputs = inputs.reshape((-1, num_inputs), order='F')
        return(inputs)

    if num_inputs == 24:
        scn.load(['confidence_an'])
        confidence = np.nan_to_num(scn['confidence_an'].values)
        inputs = np.array([S1, S2, S3, S4, S5, S6, S7,
                           S8, S9, salza, solza, lat, lon])
        confidence_flags = bits_from_int(confidence)

        inputs = np.vstack((inputs, confidence_flags))
        inputs = np.swapaxes(inputs, 0, 2)
        inputs = inputs.reshape((-1, num_inputs), order='F')

        return(inputs)


def prep_data(pixel_info, bayesian=False, cnn=False, seed=None):
    """
    Prepares data for matched SLSTR and CALIOP pixels into training data,
    validation data, training truth data, validation truth data.
    Optionally ouputs bayes values for the validation set only.
    Optionally outputs sub-images for the CNN for the both the validation and
    training set.
    """
    if seed == None:
        seed = np.random.randint(0, 2**32, dtype='uint32')
        np.random.seed(seed)
    else:
        print("Using predefined seed")
        np.random.seed(seed)

    conv_pixels = pixel_info.astype(float)
    pix = np.nan_to_num(conv_pixels)

    # turns surfacetype bitmask into one-hot encoding
    pix = surftype_processing(pix)

    # seperate validation from training data
    pct = int(len(pix)*.15)
    training = pix[:-pct]   # take all but the 15% last
    validation = pix[-pct:]   # take the last 15% of pixels

    # shuffle
    np.random.shuffle(training)
    np.random.shuffle(validation)

    if bayesian is False:
        training_data = training[:, :-2]
        training_truth_flags = training[:, -2]
        validation_data = validation[:, :-2]
        validation_truth_flags = validation[:, -2]

    if bayesian is True:
        training_data = training[:, :-3]
        training_truth_flags = training[:, -2]
        validation_data = validation[:, :-3]
        validation_truth_flags = validation[:, -2]
        bayes_values = validation[:, -3]

    training_truth = []
    validation_truth = []

    for d in training_truth_flags:
        i = DL.vfm_feature_flags(int(d))
        if i == 2:
            training_truth.append([1., 0.])    # cloud
        if i != 2:
            training_truth.append([0., 1.])    # not cloud

    for d in validation_truth_flags:
        i = DL.vfm_feature_flags(int(d))
        if i == 2:
            validation_truth.append([1., 0.])    # cloud
        if i != 2:
            validation_truth.append([0., 1.])    # not cloud

    training_truth = np.array(training_truth)
    validation_truth = np.array(validation_truth)

    return_list = [training_data, validation_data, training_truth,
                   validation_truth]

    if bayesian is True:
        return_list.append([bayes_values])

    # saving the data

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open('Temp/NumpySeeds.txt', 'a') as file:
        file.write(timestamp + ': ' + str(seed) + '\n')

    return return_list


def surftype_class(array):
    """
    Input: array of matched pixel information
    Output: arrays of matched pixel information for each surface type

    Assumes that the surftype_processing has alredy been applied the matched
    pixel information array. This function is specifically used in the
    acc_stype_test.py script
    """

    coastline = []
    ocean = []
    tidal = []
    land = []
    inland_water = []
    unfilled = []
    spare1 = []
    spare2 = []
    cosmetic = []
    duplicate = []
    day = []
    twilight = []
    sun_glint = []
    snow = []
    summary_cloud = []
    summary_pointing = []

    # sorting data point into surface type categories from the one-hot encoding
    # added in the previous step

    for d in array:
        if int(d[13]) == 1:
            coastline.append(d)
        if int(d[14]) == 1:
            ocean.append(d)
        if int(d[15]) == 1:
            tidal.append(d)
        if int(d[16]) == 1:
            land.append(d)
        if int(d[17]) == 1:
            inland_water.append(d)
        if int(d[18]) == 1:
            unfilled.append(d)
        if int(d[19]) == 1:
            spare1.append(d)
        if int(d[20]) == 1:
            spare2.append(d)
        if int(d[21]) == 1:
            cosmetic.append(d)
        if int(d[22]) == 1:
            duplicate.append(d)
        if int(d[23]) == 1:
            day.append(d)
        if int(d[24]) == 1:
            twilight.append(d)
        if int(d[25]) == 1:
            sun_glint.append(d)
        if int(d[26]) == 1:
            snow.append(d)
        if int(d[27]) == 1:
            summary_cloud.append(d)
        if int(d[28]) == 1:
            summary_pointing.append(d)

    coastline = np.array(coastline)
    ocean = np.array(ocean)
    tidal = np.array(tidal)
    land = np.array(land)
    inland_water = np.array(inland_water)
    unfilled = np.array(unfilled)
    spare1 = np.array(spare1)
    spare2 = np.array(spare2)
    cosmetic = np.array(cosmetic)
    duplicate = np.array(duplicate)
    day = np.array(day)
    twilight = np.array(twilight)
    sun_glint = np.array(sun_glint)
    snow = np.array(snow)
    summary_cloud = np.array(summary_cloud)
    summary_pointing = np.array(summary_pointing)

    # the output is ready to be fed into a for loop to calculate model accuracy
    # as a function of surface type

    return [coastline, ocean, tidal, land, inland_water, unfilled, spare1,
            spare2, cosmetic, duplicate, day, twilight, sun_glint, snow,
            summary_cloud, summary_pointing]


def bits_from_int(array):
    array = array.astype(int)
    coastline = array & 1
    ocean = array & 2
    tidal = array & 4
    land = array & 8
    inland_water = array & 16
    cosmetic = array & 256
    duplicate = array & 512
    day = array & 1024
    twilight = array & 2048
    sun_glint = array & 4096
    snow = array & 8192
    out = np.array([coastline, ocean, tidal, land, inland_water, cosmetic,
                    duplicate, day, twilight, sun_glint, snow])
    out = (out > 0).astype(int)
    return(out)


def surftype_processing(array):
    """
    Bitwise processing of SLSTR surface data. The different surface types are :
    1: coastline
    2: ocean
    4: tidal
    8: land
    16: inland_water
    32: unfilled            -
    64: spare               -
    128: spare              -
    256: cosmetic
    512: duplicate
    1024: day
    2048: twilight
    4096: sun_glint
    8192: snow
    16384: summary_cloud    -
    32768: summary_pointing -

    Input: array of matched pixel information
    Output: array of matched pixel information with processed surface type (one
    hot encoded)
    """

    # sorting data point into surface type categories using bitwise addition

    surftype_list = []

    for d in array:
        confidence = d[13]
        bitmask = format(int(confidence), '#018b')
        desired_bitmask = bitmask[4:5] + bitmask[6:10] + bitmask[13:]
        a = np.array([i for i in desired_bitmask])
        a = a.astype(int)
        surftype_list.append(a)

    surftype_list = np.array(surftype_list)

    # the new array is created by taking the first 13 values of the array
    # stiching the ones and zeros for the different surface types and then
    # linking the final two values

    if len(array) > 14:
        new_array = np.column_stack((array[:, :13], surftype_list,
                                     array[:, 14:]))
    else:
        new_array = np.column_stack((array[:, :13], surftype_list))

    return new_array
