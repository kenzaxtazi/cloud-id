
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


def getinputsFFN(Sreference, input_type=24):
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
    L4 = PixelLoader(location_directory)
    # Load one month from context dataframe
    C4 = PixelLoader(context_directory)

    c4 = C4[['Pos', 'Sfilename', 'Star_array']].values

    stars = c4[:, 2]
    padded_stars = star_padding(stars)

    print('matching datasets')

    Cpos = C4['Pos'].values
    CRows = [i[0] for i in Cpos]
    CCols = [i[1] for i in Cpos]

    C4['RowIndex'] = CRows
    C4['ColIndex'] = CCols

    merged = pd.merge(L4, C4, on=['Sfilename', 'RowIndex', 'ColIndex'])

    merged = merged.sample(frac=1)

    truth = merged['Feature_Classification_Flags'].values

    # split data into validation and training

    pct = int(len(padded_stars) * validation_frac)

    # take all but the 15% last
    training_data = padded_stars[:-pct]
    # take the last 15% of pixels
    validation_data = padded_stars[-pct:]
    training_truth_flags = truth[:-pct]
    validation_truth_flags = truth[-pct:]

    # turn binary truth flags into one hot code
    training_cloudtruth = (training_truth_flags.astype(int) & 2) / 2
    reverse_training_cloudtruth = 1 - training_cloudtruth
    training_truth = np.vstack(
        (training_cloudtruth, reverse_training_cloudtruth)).T

    validation_cloudtruth = (validation_truth_flags.astype(int) & 2) / 2
    reverse_validation_cloudtruth = 1 - validation_cloudtruth
    validation_truth = np.vstack(
        (validation_cloudtruth, reverse_validation_cloudtruth)).T

    return training_data, validation_data, training_truth, validation_truth


def cnn_getinputs(Sreference, positions=None):
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
    S1 = np.nan_to_num(scn['S1_an'].values)  # @TODO: Use or remove

    if positions is None:
        row = np.repeat(np.arange(2400), 3000)
        column = np.tile(np.arange(3000), 2400)

        star_coords = get_coords(row, column, contextlength=50)
    else:
        star_coords = get_coords(
            positions[:, 0], positions[:, 1], contextlength=50)

    star = S1[star_coords]

    return star


def surftype_class(validation_data, validation_truth, stypes, bmask, emask,
                   stypes_excluded=['summary_pointing', 'summary_cloud', 'sun_glint',
                                    'unfilled', 'spare1', 'spare2']):
    """
    Input: array of matched pixel information
    Output: arrays of matched pixel information for each surface type

    Assumes that the surftype_processing has alredy been applied the matched
    pixel information array. This function is specifically used in the
    acc_stype_test.py script
    """

    flag_names = ['coastline', 'ocean', 'tidal', 'land', 'inland_water', 'unfilled',
                  'spare1', 'spare2', 'cosmetic', 'duplicate', 'day', 'twilight',
                  'sun_glint', 'snow', 'summary_cloud', 'summary_pointing']

    if len(stypes_excluded) > 0:
        indices_to_exclude = [flag_names.index(x) for x in stypes_excluded]

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

    for i in range(len(validation_data)):
        if int(stypes[i, 0]) == 1:
            coastline.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 1]) == 1:
            ocean.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 2]) == 1:
            tidal.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 3]) == 1:
            land.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 4]) == 1:
            inland_water.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 5]) == 1:
            unfilled.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 6]) == 1:
            spare1.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 7]) == 1:
            spare2.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 8]) == 1:
            cosmetic.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 9]) == 1:
            duplicate.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 10]) == 1:
            day.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 11]) == 1:
            twilight.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 12]) == 1:
            sun_glint.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 13]) == 1:
            snow.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 14]) == 1:
            summary_cloud.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])
        if int(stypes[i, 15]) == 1:
            summary_pointing.append(
                [validation_data[i], validation_truth[i], bmask[i], emask[i]])

    coastline = np.concatenate(coastline).reshape(-1, 4)
    ocean = np.concatenate(ocean).reshape(-1, 4)
    tidal = np.concatenate(tidal).reshape(-1, 4)
    land = np.concatenate(land).reshape(-1, 4)
    inland_water = np.concatenate(inland_water).reshape(-1, 4)
    unfilled = np.concatenate(unfilled).reshape(-1, 4)
    spare1 = np.concatenate(spare1).reshape(-1, 4)
    spare2 = np.concatenate(spare2).reshape(-1, 4)
    cosmetic = np.concatenate(cosmetic).reshape(-1, 4)
    duplicate = np.concatenate(duplicate).reshape(-1, 4)
    day = np.concatenate(day).reshape(-1, 4)
    twilight = np.concatenate(twilight).reshape(-1, 4)
    sun_glint = np.concatenate(sun_glint).reshape(-1, 4)
    snow = np.concatenate(snow).reshape(-1, 4)
    summary_cloud = np.concatenate(summary_cloud).reshape(-1, 4)
    summary_pointing = np.concatenate(summary_pointing).reshape(-1, 4)

    # the output is ready to be fed into a for loop to calculate model accuracy
    # as a function of surface type

    stype_list = [coastline, ocean, tidal, land, inland_water, unfilled, spare1, spare2, cosmetic,
                  duplicate, day, twilight, sun_glint, snow, summary_cloud, summary_pointing]

    new_list = stype_list.pop(indices_to_exclude)

    return new_list


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


def star_padding(stars):
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
    padded_stars = []

    print('padding stars')

    for star in tqdm(stars):

        padded_star = []

        for arm in star:
            if len(arm) < 50:
                if len(arm) == 0:
                    # Should probably be adjacent pixel value
                    padded_arm = np.array([0] * 50)
                else:
                    padded_arm = np.pad(arm, (0, 50 - len(arm)), mode='edge')
            else:
                padded_arm = arm

            if np.mean(np.isnan(padded_arm)) == 1:
                padded_star.append(np.array([0] * 50))

            else:
                ok = ~np.isnan(padded_arm)
                xp = ok.nonzero()[0]
                fp = padded_arm[~np.isnan(padded_arm)]
                x = np.isnan(padded_arm).nonzero()[0]

                padded_arm[np.isnan(padded_arm)] = np.interp(x, xp, fp)

                padded_arm[padded_arm < 0] = 0

                padded_star.append(padded_arm)

        padded_stars.append(padded_star)

    padded_stars = np.concatenate(padded_stars).reshape((-1, 8, 50, 1))

    return padded_stars

# Class to add useful methods to pd DataFrame


@pd.api.extensions.register_dataframe_accessor("dp")
class DataPreparer():
    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self._obj = pandas_obj

    def mask_negative(self):
        Data = ['S1_an', 'S2_an', 'S3_an', 'S4_an',
                'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in']
        for col in Data:
            self._obj[col][self._obj[col] < 0] = 0

    def remove_nan(self):
        if 'confidence_in' in self._obj.columns:
            self._obj = self._obj.drop(['confidence_in'], axis=1)
        self._obj = self._obj.dropna()
        return(self._obj)

    def remove_anomalous(self, MaxDist=500, MaxTime=1200):
        self._obj = self._obj[self._obj['Distance'] < MaxDist]
        self._obj = self._obj[abs(self._obj['TimeDiff']) < MaxTime]
        return(self._obj)

    def remove_night(self):
        self._obj = self._obj[self._obj['confidence_an'] & 1024 == 1024]
        return(self._obj)

    def prepare_random(self, seed):
        if seed is None:
            seed = np.random.randint(0, 2**32, dtype='uint32')
            np.random.seed(seed)
        else:
            print("Using predefined seed")
            np.random.seed(seed)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        with open('Temp/NumpySeeds.txt', 'a') as file:
            file.write(timestamp + ': ' + str(seed) + '\n')

    def shuffle_random(self, seed=None):
        self.prepare_random(seed)
        self._obj = self._obj.sample(frac=1)
        return(self._obj)

    def shuffle_by_file(self, seed=None):
        self.prepare_random(seed)
        Sfiles = list(set(self._obj['Sfilename']))
        np.random.shuffle(Sfiles)
        sorterindex = dict(zip(Sfiles, range(len(Sfiles))))
        self._obj['Temp'] = self._obj['Sfilename'].map(sorterindex)
        self._obj = self._obj.sort_values(['Temp'])
        self._obj = self._obj.drop(['Temp'], axis=1)

    def get_ffn_training_data(self, input_type=22, validation_frac=0.15, seed=None):
        self.mask_negative()
        self.remove_nan()
        self.remove_anomalous()
        self.shuffle_by_file(seed)
        self.remove_night()

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

        pct = int(len(pix) * validation_frac)
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

    def get_cnn_training_data(self, validation_frac=0.15, seed=None):
        self.remove_nan()
        self.remove_anomalous()
        self.shuffle_by_file(seed)
        self.remove_night()

        stars = self._obj['Star_array'].values
        padded_stars = star_padding(stars)

        truth = self._obj['Feature_Classification_Flags'].values

        # split data into validation and training

        pct = int(len(padded_stars) * validation_frac)

        # take all but the 15% last
        training_data = padded_stars[:-pct]
        # take the last 15% of pixels
        validation_data = padded_stars[-pct:]
        training_truth_flags = truth[:-pct]
        validation_truth_flags = truth[-pct:]

        # turn binary truth flags into one hot code
        training_cloudtruth = (training_truth_flags.astype(int) & 2) / 2
        reverse_training_cloudtruth = 1 - training_cloudtruth
        training_truth = np.vstack(
            (training_cloudtruth, reverse_training_cloudtruth)).T

        validation_cloudtruth = (validation_truth_flags.astype(int) & 2) / 2
        reverse_validation_cloudtruth = 1 - validation_cloudtruth
        validation_truth = np.vstack(
            (validation_cloudtruth, reverse_validation_cloudtruth)).T

        return training_data, validation_data, training_truth, validation_truth

    def get_inputs(self, input_type=24):

        pixel_channels = (self._obj[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in',
                                     'satellite_zenith_angle', 'solar_zenith_angle', 'latitude_an', 'longitude_an']].values).astype(float)

        confidence_ints = self._obj['confidence_an'].values

        confidence_flags = bits_from_int(confidence_ints, input_type)

        confidence_flags = confidence_flags.T

        inputs = np.column_stack((pixel_channels, confidence_flags))

        return(inputs)

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
        self._obj['CTruth'] = pd.Series(CTruth, index=self._obj.index)
        return(self._obj)

    def make_STruth_col(self, cloudmask='cloud_an', bit=1):
        bitfield = self._obj[cloudmask]
        val = bitfield.astype(int)
        val = val & bit
        STruth = val == bit
        self._obj['STruth'] = pd.Series(STruth, index=self._obj.index)
        return(self._obj)
