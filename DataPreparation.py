
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

    inputs = np.array([
        np.nan_to_num(scn['S1_an'].values),
        np.nan_to_num(scn['S2_an'].values),
        np.nan_to_num(scn['S3_an'].values),
        np.nan_to_num(scn['S4_an'].values),
        np.nan_to_num(scn['S5_an'].values),
        np.nan_to_num(scn['S6_an'].values),
        DL.upscale_repeat(np.nan_to_num(scn['S7_in'].values)),
        DL.upscale_repeat(np.nan_to_num(scn['S8_in'].values)),
        DL.upscale_repeat(np.nan_to_num(scn['S9_in'].values)),
        DL.upscale_repeat(np.nan_to_num(scn['satellite_zenith_angle'].values)),
        DL.upscale_repeat(np.nan_to_num(scn['solar_zenith_angle'].values)),
        np.nan_to_num(scn['latitude_an'].values),
        np.nan_to_num(scn['longitude_an'].values)
    ])

    if input_type == 13:
        inputs = np.reshape(inputs, (13, 7200000))
        return(inputs.T)

    scn.load(['confidence_an'])
    confidence = np.nan_to_num(scn['confidence_an'].values)
    confidence_flags = bits_from_int(confidence, input_type)

    inputs = np.vstack((inputs, confidence_flags))

    if input_type == 21:
        inputs = np.reshape(inputs, (21, 7200000))

    if input_type == 22:
        inputs = np.reshape(inputs, (22, 7200000))

    elif input_type == 24:
        inputs = np.reshape(inputs, (24, 7200000))

    return(inputs.T)


def getinputsCNN(Sreference, indices):
    row = (indices / 3000).astype(int)
    col = (indices % 3000).astype(int)
    if type(Sreference) == str:
        scn = DL.scene_loader(Sreference)
    else:
        scn = Sreference

    scn.load(['S1_an'])
    S1 = np.nan_to_num(scn['S1_an'].values)
    data = []

    for i in range(len(row)):
        coords = get_coords(row[i], col[i], 50, True)
        star = []
        for arm in coords:
            if len(arm) > 0:
                arm = np.array(arm)
                arm_row = arm[:, 0]
                arm_col = arm[:, 1]
                arm_data = S1[arm_row, arm_col]
                star.append(arm_data)
            else:
                star.append([])
        data.append(star)

    return data

def bits_from_int(array, num_inputs=22):
    """
    Extract bitmask from integer arrays

    Parameters
    ----------
    array: array
        Raw values of confidence_an data from an SLSTR file

    num_inputs: int, either 22 or 24
        Total number of inputs for FFN model being used
        Default is 22

    Returns
    ----------
    out: array, shape ``(-1, num_inputs - 13)``
        Boolean mask where each row shows which of the confidence_an
        surface type flags are set for each pixel
    """
    array = array.astype(int)

    if num_inputs == 21:
        return(
            np.array([
                array & 1,          # Coastline
                array & 2,          # Ocean
                array & 4,          # Tidal
                array & 24 == 8,    # Dry land
                array & 16,         # Inland water
                array & 256,        # Cosmetic
                array & 512,        # Duplicate
                array & 1024,       # Day
            ]).astype('bool')
        )

    if num_inputs == 22:
        return(
            np.array([
                array & 1,          # Coastline
                array & 2,          # Ocean
                array & 4,          # Tidal
                array & 24 == 8,    # Dry land
                array & 16,         # Inland water
                array & 256,        # Cosmetic
                array & 512,        # Duplicate
                array & 1024,       # Day
                array & 2048        # Twilight
            ]).astype('bool')
        )

    if num_inputs == 24:
        return(
            np.array([
                array & 1,          # Coastline
                array & 2,          # Ocean
                array & 4,          # Tidal
                array & 8,          # Land
                array & 16,         # Inland water
                array & 256,        # Cosmetic
                array & 512,        # Duplicate
                array & 1024,       # Day
                array & 2048,       # Twilight
                array & 4096,       # Sun glint
                array & 8192,       # Snow
            ]).astype('bool')
        )

    else:
        raise ValueError('Only recognised num_inputs values are 21, 22 and 24')


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
        self.seed = np.random.randint(0, 2**32, dtype='uint32')

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
            np.random.seed(self.seed)
        else:
            print("Using predefined seed")
            self.seed = seed
            np.random.seed(seed)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        with open('Temp/NumpySeeds.txt', 'a') as file:
            file.write(timestamp + ': ' + str(self.seed) + '\n')

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
        # self.remove_night()

        pixel_inputs = self.get_inputs(input_type)

        pixel_outputs = self._obj[['Feature_Classification_Flags']].values

        pix = np.column_stack((pixel_inputs, pixel_outputs)).astype('float')

        pct = int(len(pix) * validation_frac)

        training_data = pix[:-pct, :input_type]
        training_truth_flags = pix[:-pct, input_type]

        validation_data = pix[-pct:, :input_type]
        validation_truth_flags = pix[-pct:, input_type]

        training_cloudtruth = (training_truth_flags.astype(int) & 7 == 2)
        training_truth = np.vstack(
            (training_cloudtruth, ~training_cloudtruth)).T

        validation_cloudtruth = (validation_truth_flags.astype(int) & 7 == 2)
        validation_truth = np.vstack(
            (validation_cloudtruth, ~validation_cloudtruth)).T

        return training_data, validation_data, training_truth, validation_truth

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
        training_cloudtruth = (training_truth_flags.astype(int) & 7 == 2)
        training_truth = np.vstack(
            (training_cloudtruth, ~training_cloudtruth)).T

        validation_cloudtruth = (validation_truth_flags.astype(int) & 7 == 2)
        validation_truth = np.vstack(
            (validation_cloudtruth, ~validation_cloudtruth)).T

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
