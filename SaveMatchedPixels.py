# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 23:51:44 2018

@author: tomzh
"""

import DataLoader as DL
import CalipsoReader2 as CR
from Collocation2 import collocate
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools


def get_file_pairs():
    # Open Matches.txt and return path to pairs
    caliop_directory = '/home/hep/trz15/cloud/Calipso/1km/2018/04'
    slstr_directory = '/home/hep/trz15/cloud/SLSTR/2018/04'
    with open('Matches.txt', 'r') as file:
        data = file.readlines()

    num_pairs = len(data)
    Cfilenames = []
    Sfilenames = []
    for i in data:
        pairings = i.split(',')
        Cfilenames.append(pairings[0])
        Sfilenames.append(pairings[1])

    Cpaths = []
    Spaths = []
    for i in range(num_pairs):
        day = Cfilenames[i][43:45]
        Cpaths.append(caliop_directory + '/' + day + '/' + Cfilenames[i])

        Spaths.append(slstr_directory + '/' + Sfilenames[i] + '.SEN3')

    return(Cpaths, Spaths)


def process_all(Spaths, Cpaths):
    num_files = len(Spaths)
    df = pd.DataFrame()
    for i in tqdm(range(num_files)):
        newdf = make_df(Spaths[i], Cpaths[i])
        if newdf != None:
            df = df.append(newdf, ignore_index=True)
        if i % 10 == 0:
            df.to_pickle('April.pkl')
    return(df)

def make_df(Spath, Cpath):
    # Make a pandas dataframe for a single file pair
    df = pd.DataFrame()

    coords = collocate(Spath, Cpath)
    if coords == None:
        return(None)
    rows = [int(i[0]) for i in coords]
    cols = [int(i[1]) for i in coords]
    Cindices = [int(i[2]) for i in coords]
    num_values = len(rows)
    scn = DL.scene_loader(Spath)
    SLSTR_attributes = ['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in', 'bayes_an', 'bayes_bn', 'bayes_cn',
                        'bayes_in', 'cloud_an', 'cloud_bn', 'cloud_cn', 'cloud_in', 'satellite_zenith_angle', 'solar_zenith_angle', 'latitude_an', 'longitude_an']
    scn.load(SLSTR_attributes)

    def Smake_series(Sattribute):

        hrows = [int(i/2) for i in rows]
        hcols = [int(i/2) for i in cols]
        if Sattribute in ['S7_in', 'S8_in', 'S9_in', 'bayes_in', 'cloud_in', 'satellite_zenith_angle', 'solar_zenith_angle']:
            data = scn[Sattribute].values[hrows, hcols]
        else:
            data = scn[Sattribute].values[rows, cols]
        return(pd.Series(data, name=Sattribute))

    with CR.SDopener(Cpath) as file:
        Feature_Classification_Flags = CR.load_data(
            file, 'Feature_Classification_Flags')
        Latitude = CR.load_data(file, 'Latitude')
        Longitude = CR.load_data(file, 'Longitude')
        Profile_Time = CR.load_data(file, 'Profile_Time')
        Solar_Zenith_Angle = CR.load_data(file, 'Solar_Zenith_Angle')

    Calipso_attributes = [Feature_Classification_Flags,
                          Latitude, Longitude, Profile_Time, Solar_Zenith_Angle]
    Calipso_attribute_names = ['Feature_Classification_Flags',
                               'Latitude', 'Longitude', 'Profile_Time', 'Solar_Zenith_Angle']

    def Cmake_series(Cattribute):
        if np.shape(Cattribute)[1] != 1:
            out = Cattribute[:, 0][Cindices]
        else:
            out = Cattribute[Cindices]
            if type(out[0]) == np.ndarray:
                out = list(itertools.chain.from_iterable(out))
        return(pd.Series(out, name=str(Cattribute)))

    for attribute in SLSTR_attributes:
        df = df.append(Smake_series(attribute))

    for attribute in Calipso_attributes:
        df = df.append(Cmake_series(attribute))

    Sfilenameser = pd.Series([Spath[-99:]] * num_values, name='Sfilename')
    Cfilenameser = pd.Series([Cpath[-60:]] * num_values, name='Cfilename')
    
    df = df.append(Sfilenameser)
    df = df.append(Cfilenameser)
    df = df.transpose()
    df.columns = SLSTR_attributes + Calipso_attribute_names + ['Sfilename', 'Cfilename']
    return(df)

if __name__ == "__main__":
    Cpaths, Spaths = get_file_pairs()
    df = process_all(Spaths, Cpaths)
    
    