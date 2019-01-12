# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 23:51:44 2018

@author: tomzh
"""

from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm

import CalipsoReader2 as CR
import DataLoader as DL
from Collocation2 import collocate


def grid_interpolate(q0, q1):
    """Find all points on a grid which lie between two points"""
    out = []
    row0, col0 = q0
    row1, col1 = q1

    if row0 - row1 == 0:
        row = row0
        for col in range(min(col0, col1) + 1, max(col0, col1)):
            out.append([row, col])
    elif col0 - col1 == 0:
        col = col0
        for row in range(min(row0, row1) + 1, max(row0, row1)):
            out.append([row, col])
    else:
        for row in range(min(row0, row1), max(row0, row1) + 1):
            for col in range(min(col0, col1), max(col0, col1) + 1):
                truecol = col0 + ((col1 - col0)/(row1 - row0)) * (row - row0)
                if abs(truecol - col) <= 0.5:
                    out.append([row, col])
    return(out[1:-1])


def get_file_pairs(slstr_directory, matchesfile, failed_downloads=[], caliop_directory="", CATS_directory=""):
    """Open matches file and return path to included data files, exclude files which were not downloaded"""
    with open(matchesfile, 'r') as file:
        data = file.readlines()

    Cfilenames = []
    Sfilenames = []
    for i in data:
        pairings = i.split(',')
        if pairings[1] not in failed_downloads:
            Cfilenames.append(pairings[0])
            Sfilenames.append(pairings[1])

    num_pairs = len(Sfilenames)

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
        newdf = process_pair(Spaths[i], Cpaths[i])
        if len(newdf) != 0:
            df = df.append(newdf, ignore_index=True, sort=True)
        if i % 10 == 0:
            df.to_pickle(pkl_output_name)
    return(df)


def process_pair(Spath, Cpath, interpolate=True):
    """Make a pandas dataframe for a given SLSTR and Calipso/CATS file pair"""
    # Find collocated pixels
    coords = collocate(Spath, Cpath)
    if coords == None:
        return(pd.DataFrame())
    rows = [int(i[0]) for i in coords]
    cols = [int(i[1]) for i in coords]
    Cindices = [int(i[2]) for i in coords]

    if interpolate is True:
        coords1 = []
        if Cpath.endswith('f'):
            with CR.SDopener(Cpath) as file:
                Feature_Classification_Flags = CR.load_data(
                    file, 'Feature_Classification_Flags')[:, 0].flatten()
            FCF = Feature_Classification_Flags[Cindices] & 7
            Ctruth = np.where(FCF == 2, True, False)
        elif Cpath.endswith('5'):
            file = h5py.File(Cpath)
            Sky_Condition_Fore_FOV = np.array(
                file['layer_descriptor']['Sky_Condition_Fore_FOV'])
            SC = Sky_Condition_Fore_FOV[Cindices]
            Ctruth = np.where(SC > 1, True, False)

        for i in range(len(Cindices) - 1):
            if Ctruth[i] == Ctruth[i+1] and abs(Cindices[i+1] - Cindices[i]) == 1:
                # Interpolate to find SLSTR pixels between the two already collocated
                pos0 = [rows[i], cols[i]]
                pos1 = [rows[i+1], cols[i+1]]
                Sindices1 = grid_interpolate(pos0, pos1)

                for j in Sindices1[int(len(Sindices1)/2):]:
                    coords1.append([j[0], j[1], Cindices[i]])
                for j in Sindices1[:int(len(Sindices1)/2)]:
                    coords1.append([j[0], j[1], Cindices[i+1]])

    def make_df(coords):
        if coords == None:
            return(pd.DataFrame())
        rows = [int(i[0]) for i in coords]
        cols = [int(i[1]) for i in coords]
        Cindices = [int(i[2]) for i in coords]
        num_values = len(rows)
        # Initialise output dataframe
        df = pd.DataFrame()
        # Load SLSTR data and desired attributes
        scn = DL.scene_loader(Spath)
        SLSTR_attributes = ['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in', 'bayes_an', 'bayes_bn', 'bayes_cn',
                            'bayes_in', 'cloud_an', 'cloud_bn', 'cloud_cn', 'cloud_in', 'satellite_zenith_angle', 'solar_zenith_angle', 'latitude_an', 'longitude_an', 'confidence_an']
        scn.load(SLSTR_attributes)

        def Smake_series(Sattribute):
            """Make a labelled pandas series for a given SLSTR attribute for all matched pixels in an SLSTR file"""
            # Prepare second index system for data on 1km instead of 0.5km grid
            hrows = [int(i/2) for i in rows]
            hcols = [int(i/2) for i in cols]
            if Sattribute in ['S7_in', 'S8_in', 'S9_in', 'bayes_in', 'cloud_in', 'satellite_zenith_angle', 'solar_zenith_angle']:
                data = scn[Sattribute].values[hrows, hcols]
            else:
                data = scn[Sattribute].values[rows, cols]
            return(pd.Series(data, name=Sattribute))

        # Load data from Calipso/CATS and their desired attributes
        if Cpath.endswith('f'):  # Calipso file
            with CR.SDopener(Cpath) as file:
                Feature_Classification_Flags = CR.load_data(
                    file, 'Feature_Classification_Flags')[:, 0].flatten()
                Latitude = CR.load_data(file, 'Latitude').flatten()
                Longitude = CR.load_data(file, 'Longitude').flatten()
                Profile_Time = CR.load_data(file, 'Profile_Time').flatten()
                Solar_Zenith_Angle = CR.load_data(
                    file, 'Solar_Zenith_Angle').flatten()
                IGBP_Surface_Type = CR.load_data(
                    file, 'IGBP_Surface_Type').flatten()

            Calipso_attributes = [Feature_Classification_Flags,
                                  Latitude, Longitude, Profile_Time, Solar_Zenith_Angle, IGBP_Surface_Type]
            Calipso_attribute_names = ['Feature_Classification_Flags',
                                       'Latitude', 'Longitude', 'Profile_Time', 'Solar_Zenith_Angle', 'IGBP_Surface_Type']

        elif Cpath.endswith('5'):    # CATS file
            file = h5py.File(Cpath)
            Latitude = np.array(file['geolocation']
                                ['CATS_Fore_FOV_Latitude'])[:, 1]
            Longitude = np.array(file['geolocation']
                                 ['CATS_Fore_FOV_Longitude'])[:, 1]

            Mdates = np.array(file['layer_descriptor']['Profile_UTC_Date'])
            Mtimes = np.array(file['layer_descriptor']
                              ['Profile_UTC_Time'])[:, 1]
            Mdatetimes = [datetime.strptime(str(i), "%Y%m%d") for i in Mdates]
            for i in range(len(Mdatetimes)):
                Mdatetimes[i] = Mdatetimes[i] + timedelta(days=Mtimes[i])
            Profile_Time = [i.timestamp() for i in Mdatetimes]
            Profile_Time = np.array(Profile_Time)
            Profile_Time -= 725846390.0  # Temporary workaround
            Solar_Zenith_Angle = np.array(
                file['geolocation']['Solar_Zenith_Angle'])
            Feature_Type_Fore_FOV = np.array(
                file['layer_descriptor']['Feature_Type_Fore_FOV'])[:, 0]
            Sky_Condition_Fore_FOV = np.array(
                file['layer_descriptor']['Sky_Condition_Fore_FOV'])

            CATS_attributes = [Feature_Type_Fore_FOV,
                               Latitude, Longitude, Profile_Time, Solar_Zenith_Angle, Sky_Condition_Fore_FOV]
            CATS_attribute_names = ['Feature_Type_Fore_FOV',
                                    'Latitude', 'Longitude', 'Profile_Time', 'Solar_Zenith_Angle', 'Sky_Condition_Fore_FOV']

        def Cmake_series(Cattribute):
            """Make a labelled pandas series for a given Calipso/CATS attribute for all matched pixels in an Calipso/CATS file"""
            out = Cattribute[Cindices]

            return(pd.Series(out, name=str(Cattribute)))

        # Add the attributes to the output dataframe
        for attribute in SLSTR_attributes:
            df = df.append(Smake_series(attribute))

        if Cpath.endswith('f'):  # Calipso file
            for attribute in Calipso_attributes:
                df = df.append(Cmake_series(attribute))

        elif Cpath.endswith('5'):  # CATS file
            for attribute in CATS_attributes:
                df = df.append(Cmake_series(attribute))

        # Add column with the origin filename
        Sfilenameser = pd.Series([Spath.split('/')[-1]]
                                 * num_values, name='Sfilename')
        Cfilenameser = pd.Series([Cpath.split('/')[-1]]
                                 * num_values, name='Cfilename')

        df = df.append(Sfilenameser)
        df = df.append(Cfilenameser)
        df = df.transpose()

        # Label the data columns
        if Cpath.endswith('f'):
            df.columns = SLSTR_attributes + \
                Calipso_attribute_names + ['Sfilename', 'Cfilename']
        elif Cpath.endswith('5'):
            df.columns = SLSTR_attributes + \
                CATS_attribute_names + ['Sfilename', 'Cfilename']

        return(df)

    if interpolate is False:
        return(make_df(coords))

    if interpolate is True:
        return(make_df(coords + coords1))


def add_time_col(df):
    # Prepare tqdm to work with pandas operations
    tqdm.pandas()

    def time_diff(row):
        # Calculate time between satellite measurements, +ve if Calipso before SLSTR
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
    # Prepare tqdm to work with pandas operations
    tqdm.pandas()

    def get_dist(row):
        # Calculate distance between pixels using geodesic formula
        dist = geodesic((row['latitude_an'], row['longitude_an']),
                        (row['Latitude'], row['Longitude'])).m
        return(dist)
    df['Distance'] = df.progress_apply(lambda row: get_dist(row), axis=1)
    return(df)
