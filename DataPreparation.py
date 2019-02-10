
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.utils
from tqdm import tqdm


import DataLoader as DL


def PixelLoader(directory):
    """Load all pixels in a directory of pickle files into a single DataFrame"""

    if directory.endswith('/') is False:
        directory += '/'
    PickleFilenames = os.listdir(directory)
    PicklePaths = [directory + i for i in PickleFilenames]

    out = pd.DataFrame()
    for file in PicklePaths:
        if file.endswith('.pkl'):
            out = out.append(pd.read_pickle(
                file), sort=True, ignore_index=True)

    print("%s pixels loaded" % (len(out)))
    return(out)


def getinputs(Sreference, input_type=24):
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

    if input_type == 13:
        inputs = np.array([S1, S2, S3, S4, S5, S6, S7, S8, S9, salza,
                           solza, lat, lon])
        inputs = np.reshape(inputs, (13, 7200000))
        return(inputs.T)

    scn.load(['confidence_an'])
    confidence = np.nan_to_num(scn['confidence_an'].values)
    inputs = np.array([S1, S2, S3, S4, S5, S6, S7,
                       S8, S9, salza, solza, lat, lon])
    confidence_flags = bits_from_int(confidence, input_type)

    inputs = np.vstack((inputs, confidence_flags))

    if input_type == 22:
        inputs = np.reshape(inputs, (22, 7200000))

    elif input_type == 24:
        inputs = np.reshape(inputs, (24, 7200000))

    return(inputs.T)


def pkl_prep_data(directory, input_type=24, validation_frac=0.15, bayesian=False, empirical=False, TimeDiff=False, seed=None, MaxDist=500, MaxTime=1200, NaNFilter=True):
    """
    Prepares a set of data for training the FFN

    Parameters
    -----------
    directory: string
        path to dowload pickle files from.
    validation_frac: float btw 0 and 1
        The fraction of the complete dataset that is taken as validation data.
    bayesian: boolean
        If True, outputs bayesian mask values.
    seed: int
        Random generator seed to shuffle data.
    MaxDist: int or float,
        Maximum collocation distance.
    MaxTime: int or float,
        Maximum collocation time.

    Returns
    ---------
    return_list: list
        List of 5 elements including the training data, validation data, training truth,
        validation truth and bayesian mask values or None.
    """
    # Record RNG seed to file, or set custom seed.
    if seed == None:
        seed = np.random.randint(0, 2**32, dtype='uint32')
        np.random.seed(seed)
    else:
        print("Using predefined seed")
        np.random.seed(seed)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open('Temp/NumpySeeds.txt', 'a') as file:
        file.write(timestamp + ': ' + str(seed) + '\n')

    # Load collocated pixels from dataframe
    df = PixelLoader(directory)

    # Remove high timediff / distance pixels ...
    df = df[df['Distance'] < MaxDist]
    df = df[abs(df['TimeDiff']) < MaxTime]

    if NaNFilter is True:
        # Remove pixels where channels have NAN values
        # S4 channel is most likely to have a NAN value
        df = df.drop(['confidence_in'], axis=1)
        df = df.dropna()

    # TODO need to shuffle after sperating data
    pixels = sklearn.utils.shuffle(df, random_state=seed)

    confidence_int = pixels['confidence_an'].values

    confidence_flags = bits_from_int(confidence_int, input_type)

    confidence_flags = confidence_flags.T

    pixel_indices = pixels.index.values

    pixel_channels = (pixels[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an',
                              'S7_in', 'S8_in', 'S9_in', 'satellite_zenith_angle',
                              'solar_zenith_angle', 'latitude_an', 'longitude_an']].values).astype(float)
    pixel_channels = np.nan_to_num(pixel_channels)

    pixel_inputs = np.column_stack((pixel_channels, confidence_flags))

    pixel_outputs = pixels[[
        'Feature_Classification_Flags', 'bayes_in', 'cloud_an', 'TimeDiff']].values

    pix = np.column_stack((pixel_inputs, pixel_outputs))
    pix = np.column_stack((pix, pixel_indices))

    pix = pix.astype(float)

    pct = int(len(pix)*validation_frac)
    training = pix[:-pct, :]   # take all but the 15% last
    validation = pix[-pct:, :]   # take the last 15% of pixels

    training_data = training[:, :24]
    training_truth_flags = training[:, 24]
    validation_data = validation[:, :24]
    validation_truth_flags = validation[:, 24]

    if bayesian is True:
        bayes_values = validation[:, 25]
    else:
        bayes_values = None

    if empirical is True:
        empirical_values = validation[:, 26]
    else:
        empirical_values = None

    if TimeDiff is True:
        times = validation[:, -2]
    else:
        times = None

    training_cloudtruth = (training_truth_flags.astype(int) & 2) / 2
    reverse_training_cloudtruth = 1 - training_cloudtruth
    training_truth = np.vstack(
        (training_cloudtruth, reverse_training_cloudtruth)).T

    validation_cloudtruth = (validation_truth_flags.astype(int) & 2) / 2
    reverse_validation_cloudtruth = 1 - validation_cloudtruth
    validation_truth = np.vstack(
        (validation_cloudtruth, reverse_validation_cloudtruth)).T

    return_list = [training_data, validation_data, training_truth,
                   validation_truth, bayes_values, empirical_values, times]
    return return_list


def cnn_prep_data(location_directory, context_directory, validation_frac=0.15):
    """
    Prepares data for matched SLSTR and CALIOP pixels into training data,
    validation data, training truth data, validation truth data for the supermodel.
    Optionally ouputs bayes values for the validation set only.

    Parameters
    -----------
    location_directory
        direction with the pixel locations and truths

    context_directory:
        directory with context information 

    validation_frac: float between 0 and 1
        the fraction of the dataset that is taken as validation 

    Returns
    ---------
    training_data: array 

    validation_data: array

    training_truth: array

    validation_truth: array

    """

    # Load collocated pixels from dataframe
    P4 = PixelLoader(location_directory)
    # Load one month from context dataframe
    C4 = PixelLoader(context_directory)

    p4 = P4[['RowIndex', 'ColIndex', 'Sfilename']]
    c4 = C4[['Pos', 'Sfilename', 'Star_array']]

    Sfiles = list(set(p4['Sfilename']))

    truth = P4['Feature_Classification_Flags'].values
    data = []

    for file in Sfiles:
        ldf = p4[p4['Sfilename'] == file]
        cdf = c4[c4['Sfilename'] == file]

        ldf = ldf.values
        cdf = cdf.values

        for i in ldf:
            print(((cdf[:, 0])[0])[0])
            star_row = cdf[((cdf[:, 0])[0])[0] == i[0]]
            if len(star_row) > 0:
                star_column = star_row[((star_row[:, 0])[0])[1] == i[1]]
                if len(star_column) > 0:
                    star = star_column[2]
                    padded_star = star_padding(star)
                    data.append(padded_star)

    data = np.array(data)

    pct = int(len(data)*validation_frac)
    training_data = data[:-pct, :]   # take all but the 15% last
    validation_data = data[-pct:, :]   # take the last 15% of pixels
    training_truth_flags = truth[:-pct, :]
    validation_truth_flags = truth[-pct:, :]

    training_cloudtruth = (training_truth_flags.astype(int) & 2) / 2
    reverse_training_cloudtruth = 1 - training_cloudtruth
    training_truth = np.vstack(
        (training_cloudtruth, reverse_training_cloudtruth)).T

    validation_cloudtruth = (validation_truth_flags.astype(int) & 2) / 2
    reverse_validation_cloudtruth = 1 - validation_cloudtruth
    validation_truth = np.vstack(
        (validation_cloudtruth, reverse_validation_cloudtruth)).T

    return training_data, validation_data, training_truth, validation_truth


def context_getinputs(S1, Sreference, data):
    """ 
    Download and prepares pixel contextual information for a given SLSTR file to get the Supermodel prediction.

    Parameters
    -----------
    Sreferenc: string
        path to dowload SLST files from.
    data: multi dimensional array
        0: probability from first model
        1: indice from image

    Returns
    ---------
    star: array
        data for CNN
    """

    if type(Sreference) == str:
        scn = DL.scene_loader(Sreference)
    else:
        scn = Sreference

    scn.load(['S1_an'])
    S1 = np.nan_to_num(scn['S1_an'].values)

    row = int(float(data[1])/2400.)
    column = data[1] % 3000

    star = get_coords(row, column, contextlength=25)

    return star


def surftype_class(validation_data, validation_truth, masks=None):
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
    cosmetic = []
    duplicate = []
    day = []
    twilight = []
    sun_glint = []
    snow = []

    # sorting data point into surface type categories from the one-hot encoding
    # added in the previous step
    if masks is not None:
        for i in range(len(validation_data)):
            if int(validation_data[i, 13]) == 1:
                coastline.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 14]) == 1:
                ocean.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 15]) == 1:
                tidal.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 16]) == 1:
                land.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 17]) == 1:
                inland_water.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 18]) == 1:
                cosmetic.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 19]) == 1:
                duplicate.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 20]) == 1:
                day.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 21]) == 1:
                twilight.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 22]) == 1:
                sun_glint.append(np.array(
                    [validation_data[i], validation_truth[i], masks[i]]))
            if int(validation_data[i, 23]) == 1:
                snow.append(
                    np.array([validation_data[i], validation_truth[i], masks[i]]))

    if masks is None:
        for i in range(len(validation_data)):
            if int(validation_data[i, 13]) == 1:
                coastline.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 14]) == 1:
                ocean.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 15]) == 1:
                tidal.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 16]) == 1:
                land.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 17]) == 1:
                inland_water.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 18]) == 1:
                cosmetic.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 19]) == 1:
                duplicate.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 20]) == 1:
                day.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 21]) == 1:
                twilight.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 22]) == 1:
                sun_glint.append([validation_data[i], validation_truth[i]])
            if int(validation_data[i, 23]) == 1:
                snow.append([validation_data[i], validation_truth[i]])

        coastline = np.array(coastline)
        ocean = np.array(ocean)
        tidal = np.array(tidal)
        land = np.array(land)
        inland_water = np.array(inland_water)
        cosmetic = np.array(cosmetic)
        duplicate = np.array(duplicate)
        day = np.array(day)
        twilight = np.array(twilight)
        sun_glint = np.array(sun_glint)
        snow = np.array(snow)

        # the output is ready to be fed into a for loop to calculate model accuracy
        # as a function of surface type

    return [coastline, ocean, tidal, land, inland_water, cosmetic,
            duplicate, day, twilight, sun_glint, snow]


def pad_array(a, targetshape=(25, 9), padvalue=-1):
    zeros = np.zeros(targetshape)
    zeros = zeros + padvalue

    zeros[:a.shape[0], :a.shape[1]] = a
    return(zeros)

    # TODO: Optimise
    out = []

    Pos = context_df[['RowIndex', 'ColIndex']].values
    Pos = tuple(map(tuple, Pos))
    context_df['Pos'] = Pos

    Sfiles = list(set(truth_df['Sfilename']))

    for Sfile in tqdm(Sfiles):
        pix1 = truth_df[truth_df['Sfilename'] == Sfile]
        con1 = context_df[context_df['Sfilename'] == Sfile]

        RowIndices = pix1['RowIndex'].values
        ColIndices = pix1['ColIndex'].values

        for i in range(len(RowIndices)):
            x0, y0 = RowIndices[i], ColIndices[i]

            coords = get_coords(x0, y0, 25)

            df = con1[con1['Pos'].isin(coords)]
            df = df.sort_values('Pos')

            channels = ['S1_an', 'S2_an', 'S3_an', 'S4_an',
                        'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in']

            W3_df = df[df['ColIndex'] < y0]
            E3_df = df[df['ColIndex'] > y0]
            NS_df = df[df['ColIndex'] == y0]

            W_array = pad_array(
                W3_df[W3_df['RowIndex'] == x0][channels].values[::-1])
            NW_array = pad_array(
                W3_df[W3_df['RowIndex'] < x0][channels].values[::-1])
            SW_array = pad_array(
                W3_df[W3_df['RowIndex'] > x0][channels].values)

            E_array = pad_array(
                E3_df[E3_df['RowIndex'] == x0][channels].values)
            NE_array = pad_array(
                E3_df[E3_df['RowIndex'] < x0][channels].values[::-1])
            SE_array = pad_array(
                E3_df[E3_df['RowIndex'] > x0][channels].values)

            N_array = pad_array(
                NS_df[NS_df['RowIndex'] < x0][channels].values[::-1])
            S_array = pad_array(NS_df[NS_df['RowIndex'] > x0][channels].values)

            star = np.array([N_array, NE_array, E_array, SE_array,
                             S_array, SW_array, W_array, NW_array])

            out.append(star)

    return(np.array(out))


def bits_from_int(array, num_inputs=24):
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

    dry_land = land * (1 - inland_water)
    if num_inputs == 24:
        out = np.array([coastline, ocean, tidal, land, inland_water, cosmetic,
                        duplicate, day, twilight, sun_glint, snow])

    if num_inputs == 22:
        out = np.array([coastline, ocean, tidal, dry_land, inland_water, cosmetic,
                        duplicate, day, twilight])
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
        desired_bitmask = bitmask[4:6] + bitmask[6:10] + bitmask[13:]
        a = np.array([i for i in desired_bitmask])
        a = a.astype(int)
        surftype_list.append(a)

    surftype_list = np.array(surftype_list)

    # the new array is created by taking the first 13 values of the array
    # stiching the ones and zeros for the different surface types and then
    # linking the final two values
    if len(array[0]) > 14:
        new_array = np.column_stack((array[:, :13], surftype_list,
                                     array[:, 14:]))
    else:
        new_array = np.column_stack((array[:, :13], surftype_list))

    return new_array


def inputs_from_df(df, input_type=24):
    """Load values from dataframe into input array for tflearn model"""
    S1 = np.nan_to_num(df['S1_an'].values)
    S2 = np.nan_to_num(df['S2_an'].values)
    S3 = np.nan_to_num(df['S3_an'].values)
    S4 = np.nan_to_num(df['S4_an'].values)
    S5 = np.nan_to_num(df['S5_an'].values)
    S6 = np.nan_to_num(df['S6_an'].values)
    S7 = np.nan_to_num(df['S7_in'].values)
    S8 = np.nan_to_num(df['S8_in'].values)
    S9 = np.nan_to_num(df['S9_in'].values)
    salza = np.nan_to_num(df['satellite_zenith_angle'].values)
    solza = np.nan_to_num(df['solar_zenith_angle'].values)
    lat = np.nan_to_num(df['latitude_an'].values)
    lon = np.nan_to_num(df['longitude_an'].values)

    confidence = np.nan_to_num(df['confidence_an'].values)
    inputs = np.array([S1, S2, S3, S4, S5, S6, S7,
                       S8, S9, salza, solza, lat, lon])
    confidence_flags = bits_from_int(confidence, input_type)

    inputs = np.vstack((inputs, confidence_flags))

    return(inputs.T)


def get_coords(x0, y0, contextlength, separate=False):
    East_xs = np.linspace(x0 + 1, x0 + contextlength,
                          contextlength).astype(int)
    West_xs = np.linspace(x0 - 1, x0 - contextlength,
                          contextlength).astype(int)

    South_ys = np.linspace(y0 + 1, y0 + contextlength,
                           contextlength).astype(int)
    North_ys = np.linspace(y0 - 1, y0 - contextlength,
                           contextlength).astype(int)

    # Restrict to indices within 2400, 3000 frame
    East_xs = East_xs[East_xs < 2400]
    West_xs = West_xs[West_xs > -1]
    North_ys = North_ys[North_ys > -1]
    South_ys = South_ys[South_ys < 3000]

    N_list = list(zip([x0] * len(North_ys), North_ys))
    E_list = list(zip(East_xs, [y0] * len(East_xs)))
    S_list = list(zip([x0] * len(South_ys), South_ys))
    W_list = list(zip(West_xs, [y0] * len(West_xs)))

    NE_list = list(zip(East_xs, North_ys))
    SE_list = list(zip(East_xs, South_ys))
    NW_list = list(zip(West_xs, North_ys))
    SW_list = list(zip(West_xs, South_ys))

    if separate is False:
        return(N_list + NE_list + E_list + SE_list + S_list + SW_list + W_list + NW_list)
    if separate is True:
        return([N_list, NE_list, E_list, SE_list, S_list, SW_list, W_list, NW_list])


def star_padding(star):
    """
    Pads out contextual stars 

    Parameters
    -----------
    stars : array of lists 
        contextual data for a target pixel, in the shape of a star

    Returns
    ____
    padded_star: 8x50 array 
        padded contextual data for a target pixel, in the shape of a star

    """
    padded_star = []

    for arm in star:
        if len(arm) < 50:
            padded_arm = np.pad(arm, (0, 50-len(arm)),
                                mode='constant', constant_values=-5)
            padded_star.append(padded_arm)
        else:
            padded_star.append(arm)

    padded_star = np.array(padded_arm)

    return padded_star

# Class to add useful methods to pd DataFrame
@pd.api.extensions.register_dataframe_accessor("dp")
class DataPreparer():
    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self._obj = pandas_obj

    def remove_nan(self):
        if 'confidence_in' in self._obj.columns:
            self._obj = self._obj.drop(['confidence_in'], axis=1)
        self._obj = self._obj.dropna()
        return(self._obj)

    def remove_anomalous(self, MaxDist=500, MaxTime=1200):
        self._obj = self._obj[self._obj['Distance'] < MaxDist]
        self._obj = self._obj[abs(self._obj['TimeDiff']) < MaxTime]
        return(self._obj)

    def shuffle_random(self, validation_frac=0.15):
        self._obj = self._obj.sample(frac=1)
        return(self._obj)

    def shuffle_by_file(self, validation_frac=0.15):
        Sfiles = list(set(self._obj['Sfilename']))
        np.random.shuffle(Sfiles)
        sorterindex = dict(zip(Sfiles, range(len(Sfiles))))
        self._obj['Temp'] = self._obj['Sfilename'].map(sorterindex)
        self._obj = self._obj.sort_values(['Temp'])
        self._obj = self._obj.drop(['Temp'], axis=1)

    def get_training_data(self, input_type=24, validation_frac=0.15):
        self.remove_nan()
        self.remove_anomalous()
        self.shuffle_by_file(validation_frac)

        pixel_channels = (self._obj[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in',
                                     'satellite_zenith_angle', 'solar_zenith_angle', 'latitude_an', 'longitude_an']].values).astype(float)
        confidence_ints = self._obj['confidence_an'].values

        confidence_flags = bits_from_int(confidence_ints, input_type)

        confidence_flags = confidence_flags.T

        pixel_indices = self._obj.index.values

        pixel_inputs = np.column_stack((pixel_channels, confidence_flags))

        pixel_outputs = self._obj[[
            'Feature_Classification_Flags', 'bayes_in', 'cloud_an', 'TimeDiff']].values

        pix = np.column_stack((pixel_inputs, pixel_outputs))
        pix = np.column_stack((pix, pixel_indices))

        pix = pix.astype('float')

        pct = int(len(pix)*validation_frac)
        training = pix[:-pct, :]   # take all but the 15% last
        validation = pix[-pct:, :]   # take the last 15% of pixels

        training_data = training[:, :input_type]
        training_truth_flags = training[:, input_type]
        validation_data = validation[:, :input_type]
        validation_truth_flags = validation[:, input_type]

        training_cloudtruth = (training_truth_flags.astype(int) & 2) / 2
        reverse_training_cloudtruth = 1 - training_cloudtruth
        training_truth = np.vstack(
            (training_cloudtruth, reverse_training_cloudtruth)).T

        validation_cloudtruth = (validation_truth_flags.astype(int) & 2) / 2
        reverse_validation_cloudtruth = 1 - validation_cloudtruth
        validation_truth = np.vstack(
            (validation_cloudtruth, reverse_validation_cloudtruth)).T

        return_list = [training_data, validation_data, training_truth,
                       validation_truth]
        return return_list

    def get_inputs(self, input_type=24):

        pixel_channels = (self._obj[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in',
                                     'satellite_zenith_angle', 'solar_zenith_angle', 'latitude_an', 'longitude_an']].values).astype(float)

        confidence_ints = self._obj['confidence_an'].values

        confidence_flags = bits_from_int(confidence_ints, input_type)

        confidence_flags = confidence_flags.T

        pixel_inputs = np.column_stack((pixel_channels, confidence_flags))

        inputs = np.vstack((pixel_inputs, confidence_flags))

        return(inputs.T)

    def make_attrib_hist(self, column='latitude_an'):
        out = self._obj[column]
        frq, edges = np.histogram(out, 100)
        plt.title(column + ' histogram')
        plt.bar(edges[:-1], frq, width=np.diff(edges), ec='k', align='edge')
        plt.show()

    def make_CTruth_col(self):
        FCF = self._obj['Feature_Classification_Flags']
        val = FCF.astype(int)
        val = val & 7
        CTruth = val == 2
        self._obj['CTruth'] = pd.Series(CTruth, index=df.index)
        return(self._obj)

    def make_STruth_col(self, cloudmask='cloud_an', bit=1):
        bitfield = self._obj[cloudmask]
        val = bitfield.astype(int)
        val = val & bit
        STruth = val == bit
        self._obj['STruth'] = pd.Series(STruth, index=df.index)
        return(self._obj)
