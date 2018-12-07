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
from datetime import datetime, timedelta
from geopy.distance import geodesic
import h5py

def get_file_pairs(slstr_directory, matchesfile, failed_downloads=[], caliop_directory="", CATS_directory=""):
    # Open Matches.txt and return path to pairs
    with open(matchesfile, 'r') as file:
        data = file.readlines()

    num_pairs = len(data)
    Cfilenames = []
    Sfilenames = []
    for i in data:
        pairings = i.split(',')
        if pairings[1] not in failed_downloads:
            Cfilenames.append(pairings[0])
            Sfilenames.append(pairings[1])

    Cpaths = []
    Spaths = []
    if caliop_directory != "":
        for i in range(num_pairs):
            Cpaths.append(caliop_directory + '/' + Cfilenames[i])

            Spaths.append(slstr_directory + '/' + Sfilenames[i] + '.SEN3')
    elif CATS_directory != "":
        for i in range(num_pairs):
            Cpaths.append(CATS_directory + '/' + Cfilenames[i])

            Spaths.append(slstr_directory + '/' + Sfilenames[i] + '.SEN3')

    return(Cpaths, Spaths)


def process_all(Spaths, Cpaths, pkl_output_name):
    num_files = len(Spaths)
    df = pd.DataFrame()
    for i in tqdm(range(num_files)):
        newdf = make_df(Spaths[i], Cpaths[i])
        if len(newdf) != 0:
            df = df.append(newdf, ignore_index=True)
        if i % 10 == 0:
            df.to_pickle(pkl_output_name)
    return(df)


def make_df(Spath, Cpath):
    # Make a pandas dataframe for a single file pair
    df = pd.DataFrame()

    coords = collocate(Spath, Cpath)
    if coords == None:
        return(pd.DataFrame())
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

    if Cpath[0].endswith('f'):  # Calipso file
        with CR.SDopener(Cpath) as file:
            Feature_Classification_Flags = CR.load_data(
                file, 'Feature_Classification_Flags')
            Latitude = CR.load_data(file, 'Latitude')
            Longitude = CR.load_data(file, 'Longitude')
            Profile_Time = CR.load_data(file, 'Profile_Time')
            Solar_Zenith_Angle = CR.load_data(file, 'Solar_Zenith_Angle')
            IGBP_Surface_Type = CR.load_data(file, 'IGBP_Surface_Type')

        Calipso_attributes = [Feature_Classification_Flags,
                              Latitude, Longitude, Profile_Time, Solar_Zenith_Angle, IGBP_Surface_Type]
        Calipso_attribute_names = ['Feature_Classification_Flags',
                                   'Latitude', 'Longitude', 'Profile_Time', 'Solar_Zenith_Angle', 'IGBP_Surface_Type']
        def Cmake_series(Cattribute):
            if np.shape(Cattribute)[1] != 1:
                out = Cattribute[:, 0][Cindices]
            else:
                out = Cattribute[Cindices]
                if type(out[0]) == np.ndarray:
                    out = list(itertools.chain.from_iterable(out))
            return(pd.Series(out, name=str(Cattribute)))

    elif Cpath[0].endswith('5'):
        file = h5py.File(Cpath)
        Latitude = np.array(file['geolocation']['CATS_Fore_FOV_Latitude'])[:, 1]
        Longitude = np.array(file['geolocation']['CATS_Fore_FOV_Longitude'])[:, 1]

        Mdates = np.array(file['layer_descriptor']['Profile_UTC_Date'])
        Mtimes = np.array(file['layer_descriptor']['Profile_UTC_Time'])[:, 1]
        Mdatetimes = [datetime.strptime(str(i), "%Y%m%d") for i in Mdates]
        for i in range(len(Mdatetimes)):
            Mdatetimes[i] = Mdatetimes[i] + timedelta(days=Mtimes[i])
        Profile_Time = [i.timestamp() for i in Mdatetimes]
        Solar_Zenith_Angle = np.array(file['geolocation']['Solar_Zenith_Angle'])
        Feature_Type_Fore_FOV = np.array(file['layer_descriptor']['Feature_Type_Fore_FOV'])[:, 0]
        Sky_Condition_Fore_FOV = np.array(file['layer_descriptor']['Sky_Condition_Fore_FOV'])

        CATS_attributes = [Feature_Type_Fore_FOV,
                           Latitude, Longitude, Profile_Time, Solar_Zenith_Angle, Sky_Condition_Fore_FOV]
        CATS_attribute_names = ['Feature_Type_Fore_FOV',
                                'Latitude', 'Longitude', 'Profile_Time', 'Solar_Zenith_Angle', 'Sky_Condition_Fore_FOV']

        def Cmake_series(Cattribute):
            out = Cattribute[Cindices]
            if type(out[0]) == np.ndarray:
                out = list(itertools.chain.from_iterable(out))
            return(pd.Series(out, name=str(Cattribute)))


    for attribute in SLSTR_attributes:
        df = df.append(Smake_series(attribute))

    if Cpath[0].endswith('f'):
        for attribute in Calipso_attributes:
            df = df.append(Cmake_series(attribute))
    elif Cpath[0].endswith('5'):
        for attribute in CATS_attributes:
            df = df.append(Cmake_series(attribute))

    Sfilenameser = pd.Series([Spath[-99:]] * num_values, name='Sfilename')
    Cfilenameser = pd.Series([Cpath[-60:]] * num_values, name='Cfilename')

    df = df.append(Sfilenameser)
    df = df.append(Cfilenameser)
    df = df.transpose()

    if Cpath[0].endswith('f'):
        df.columns = SLSTR_attributes + \
            Calipso_attribute_names + ['Sfilename', 'Cfilename']
    elif Cpath[0].endswith('5'):
        df.columns = SLSTR_attributes + \
            CATS_attribute_names + ['Sfilename', 'Cfilename']
    return(df)


def add_time_col(df):
    tqdm.pandas()

    def time_diff(row):
        # Time between satellite measurements, +ve if Calipso before SLSTR
        Stime = row['Sfilename'][16:31]
        Stimestamp = datetime.strptime(Stime, '%Y%m%dT%H%M%S')
        Stimestamp += timedelta(minutes=1.5)
        Ctime = row['Profile_Time']
        Ctimestamp = datetime.utcfromtimestamp(Ctime)
        diff1 = Stimestamp - Ctimestamp
        if diff1.days == -1:
            diff2 = Ctimestamp - Stimestamp
            return(diff2.seconds * -1)
        else:
            return(diff1.seconds)
    df['TimeDiff'] = df.progress_apply(lambda row: time_diff(row), axis=1)

    return(df)


def add_dist_col(df):
    tqdm.pandas()

    def get_dist(row):
        dist = geodesic((row['latitude_an'], row['longitude_an']),
                        (row['Latitude'], row['Longitude'])).m
        return(dist)
    df['Distance'] = df.progress_apply(lambda row: get_dist(row), axis=1)
    return(df)


if __name__ == "__main__":
    Home_directory = "/home/hep/trz15/Masters_Project"
    NASA_FTP_directory = "8aff26d6-6b5a-4544-ac03-bdddf25d7bbb"
    calipso_directory = "/vols/lhcb/egede/cloud/Calipso/1km/2018/06/"
    SLSTR_target_directory = "/vols/lhcb/egede/cloud/SLSTR/2018/061"
    MatchesFilename = "Matches9.txt"
    pkl_output_name = "Jun.pkl"
    timewindow = 15

    Cpaths, Spaths = get_file_pairs(
        calipso_directory, SLSTR_target_directory, MatchesFilename)
    df = process_all(Spaths, Cpaths, pkl_output_name)
    df['Profile_Time'] += 725846390.0
    df = add_dist_col(df)
    df = add_time_col(df)
    processed_pkl_name = pkl_output_name[:-4] + "P1.pkl"
    df.to_pickle(processed_pkl_name)
