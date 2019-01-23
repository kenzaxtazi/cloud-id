import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from DataPreparation import bits_from_int

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


def truth_from_bitmask(row):
    val = row['Feature_Classification_Flags']
    val = np.nan_to_num(val)
    val = int(val) & 7
    if val == 2:
        return(True)
    else:
        return(False)


def make_attrib_hist(df, column='Latitude'):
    out = df[column]
    frq, edges = np.histogram(out, 100)
    plt.title(column + ' histogram')
    plt.bar(edges[:-1], frq, width=np.diff(edges), ec='k', align='edge')
    plt.show()


def make_CTruth_col(df):
    tqdm.pandas()

    def flagtruths(row):
        val = row['Feature_Classification_Flags']
        val = np.nan_to_num(val)
        val = int(val) & 7
        if val == 2:
            return(True)
        else:
            return(False)
    df['CTruth'] = df.progress_apply(lambda row: flagtruths(row), axis=1)
    return(df)


def make_STruth_col(df, cloudmask='cloud_an', bit=1):
    tqdm.pandas()

    def flagtruths(row):
        val = row[cloudmask]
        val = np.nan_to_num(val)
        val = int(val) & bit
        if val == bit:
            return(True)
        else:
            return(False)
    df['STruth'] = df.progress_apply(lambda row: flagtruths(row), axis=1)
    return(df)


def TruthMatches(df):
    q = df['CTruth'] == df['STruth']
    out = np.mean(q)
    return(out)

def inputs_from_df(df, num_inputs=24):
    """Load values from dataframe into input array for tflearn model"""
    S1 = np.nan_to_num(df['S1_an'].values)
    S2 = np.nan_to_num(df['S2_an'].values)
    S3 = np.nan_to_num(df['S3_an'].values)
    S4 = np.nan_to_num(df['S4_an'].values)
    S5 = np.nan_to_num(df['S5_an'].values)
    S6 = np.nan_to_num(df['S6_an'].values)
    S7 = (np.nan_to_num(df['S7_in'].values))
    S8 = (np.nan_to_num(df['S8_in'].values))
    S9 = (np.nan_to_num(df['S9_in'].values))
    salza = (np.nan_to_num(df['satellite_zenith_angle'].values))
    solza = (np.nan_to_num(df['solar_zenith_angle'].values))
    lat = np.nan_to_num(df['latitude_an'].values)
    lon = np.nan_to_num(df['longitude_an'].values)
    if num_inputs == 24:
        confidence = np.nan_to_num(df['confidence_an'].values)
        inputs = np.array([S1, S2, S3, S4, S5, S6, S7,
                           S8, S9, salza, solza, lat, lon])
        confidence_flags = bits_from_int(confidence)

        inputs = np.vstack((inputs, confidence_flags))
        inputs = np.swapaxes(inputs, 0, 2)
        inputs = inputs.reshape((-1, num_inputs), order='F')

        return(inputs)
    else:
        print("Invalid number of inputs")
