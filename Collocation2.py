# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:07:55 2018

@author: tomzh
"""

import io
import os
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta

import numpy as np
import requests
from tqdm import tqdm

import CalipsoReader2 as CR
import DataLoader as DL


def SLSTR_query(url):
    # Send SLSTR database query via http using default credentials
    r = requests.get(url, auth=('s3guest', 's3guest'))
    if r.status_code == 200:
        root = ET.fromstring(r.text)
        out = []
        # Number of matches:
        if 'totalResult' in str(root[5]):
            #num_matches = root[5].text
            for i in root:
                if "entry" in str(i):
                    out.append(i[0].text + "," + i[1].attrib['href'])
        else:
            print('No matches')
        return(out)

    else:
        print('Response Code Error: %s' %(r.status_code))
        return([])


def makeurlquery(Cfilename, timewindow=30, num=20):
    """Creates list of URLs to query SLSTR database for a given file"""

    # Set download website and product type
    base = "https://scihub.copernicus.eu/s3//search?q=%20producttype:SL_1_RBT___%20AND%20("

    # Load parameters from hdf file
    with CR.SDopener(Cfilename) as file:
        lat = CR.load_data(file, 'Latitude')
        lon = CR.load_data(file, 'Longitude')
        time = CR.load_data(file, 'Profile_Time')
    time += 725846400.0     # Time in UNIX
    time -= 10              # Leap second correction

    def _makequeryforslice(a, b):
        "Makes a query fragment between two calipso indices"
        c = int(0.5 * (a + b))  # Mean index

        queryfrag = "("

        # Set Time query
        timestamp = datetime.utcfromtimestamp(time[c][0])
        windowstart = timestamp - timedelta(minutes=timewindow)
        windowend = timestamp + timedelta(minutes=timewindow)
        queryfrag += "beginPosition:["
        queryfrag += str(windowstart.strftime("%Y-%m-%dT%H:%M:%S.%f"))[:-3] + 'Z'
        queryfrag += "%20TO%20"
        queryfrag += str(windowend.strftime("%Y-%m-%dT%H:%M:%S.%f"))[:-3] + 'Z' + "]"

        # Set Positional query
        queryfrag += "%20AND%20(%20footprint:%22Intersects(POLYGON(("
        queryfrag += str(lon[a][0]) + "%20" + str(lat[a][0]) + str(',')
        queryfrag += str(lon[b][0]) + "%20" + str(lat[a][0]) + str(',')
        queryfrag += str(lon[b][0]) + "%20" + str(lat[b][0]) + str(',')
        queryfrag += str(lon[a][0]) + "%20" + str(lat[b][0]) + str(',')
        queryfrag += str(lon[a][0]) + "%20" + str(lat[a][0])
        queryfrag += "%20)))%22))"

        return(queryfrag)

    # Select which indices to use to slice list
    xs = np.linspace(0, len(lat) - 1, num + 1)
    xs = xs.astype(int)

    out = []
    for i in range(num):
        if i % 10 == 0:     # Too many connected queries returns an empty response
            query = base
        else:
            query += "%20OR%20"
        query += _makequeryforslice(xs[i], xs[i+1])
        if i % 10 == 9 or i == num - 1:     # End the query
            query += ")"
            query += "&rows=25&start=0"
            out.append(query)

    return(out)


def find_SLSTR_data(Cfilename, timewindow=30, num=20):
    out = []
    queries = makeurlquery(Cfilename, timewindow, num)
    Sfilenames = []
    Sdownloads = []
    tqdm.write('Finding matches for ' + Cfilename)
    for query in queries:
        response = SLSTR_query(query)
        if response != []:
            out += response
            tqdm.write('Match found')
    out = list(set(out))
    for i in out:
        q = i.split(',')
        Sfilenames.append(q[0])
        Sdownloads.append(q[1])
    return(Sfilenames, Sdownloads)


def match_directory(directory, output='Matches.txt', timewindow=30, num=20):
    """For a directory of Calipso files, find SLSTR files which are collocated"""

    if directory[-1] != '/':
        directory += '/'

    # Find Calipso files
    q = os.listdir(directory)
    w = [i for i in q if i[-1] == 'f']

    # Query the ESA database for each file, append matches to Data and raw output file
    rawoutput = output[:-4] + "_raw.txt"
    Data = []
    for i in tqdm(range(len(w))):
        try:
            Sfilenames, Sdownloads = find_SLSTR_data(
                directory + w[i], timewindow, num)
            if Sfilenames != []:
                with open(rawoutput, 'a') as file:
                    for j in range(len(Sfilenames)):
                        file.write(
                            str(w[i]) + ',' + str(Sfilenames[j]) + ',' + str(Sdownloads[j]) + '\n')
                        Data.append([w[i], Sfilenames[j], Sdownloads[j]])
        except Exception as e:
            tqdm.write("Error: %s" %e)
            pass

    # Sort the data
    Data.sort()

    # Overwrite raw output file with sorted data
    with open(rawoutput, 'w') as file:
        for i in Data:
            file.write(i[0] + ',' + i[1] + ',' + i[2] + '\n')

    # Create new output file for unique sorted data
    uniqueoutput = output
    Sfilenames = [i[1] for i in Data]
    duplicates = []

    # Find files which are duplicates, i.e. they have the same Framenumber, Relative Orbit Number and Absolute Orbit Number
    for i in range(1, len(Sfilenames)):
        if Sfilenames[i-1][77:81] == Sfilenames[i][77:81] and Sfilenames[i-1][73:76] == Sfilenames[i][73:76] and Sfilenames[i-1][69:72] == Sfilenames[i][69:72]:
            duplicates.append(i)

    # Create unique version of sorted data
    uniquedata = []
    for i in range(len(Data)):
        if i not in duplicates:
            uniquedata.append(Data[i])

    # Output unique sorted Data to .txt
    with open(uniqueoutput, 'w') as file:
        for i in uniquedata:
            file.write(i[0] + ',' + i[1] + ',' + i[2] + '\n')
    return(Data)


def collocate(SLSTR_filename, Calipso_filename, verbose=False, persistent=False):
    # Finds pixels in both files which represent the same geographic position

    # Load SLSTR coords
    scn = DL.scene_loader(SLSTR_filename)
    scn.load(['latitude_an', 'longitude_an'])
    slat = np.array(scn['latitude_an'].values)
    slon = np.array(scn['longitude_an'].values)

    scn.unload()

    # Load Calipso coords
    with CR.SDopener(Calipso_filename) as file:
        clat = CR.load_data(file, 'Latitude')
        clon = CR.load_data(file, 'Longitude')

    # Find coord pairs which are close
    coords = []

    # Want the latitude and longitude to be within 250m of each others
    # 250m = 0.00224577793 degrees lon at equator
    # 250m = 0.00224577793 / cos(lat) degrees lon at lat
    lattolerance = 0.00224577793

    def match_SLSTR_pixel(indices):
        out = []
        i, j = indices
        try:
            matches = abs(slat[i, j] - clat) < lattolerance
            if matches.any():
                loc = np.where(matches == True)
                lontolerance = (
                    lattolerance / np.cos(slat[i, j] * np.pi / 180))
                for k in loc[0]:
                    if abs(slon[i, j] - clon[k]) < lontolerance:
                        out.append([i, j, k])
        except IndexError:
            pass
        if out == []:
            out = None
        return(out)

    def findedgepixel():
        # Check near the edge of SLSTR matrix for matches

        # Top/Bottom Row
        for i in [0, 1, 2, 3, 4, 2399, 2398, 2397, 2396, 2395]:
            for j in range(3000):
                out = match_SLSTR_pixel([i, j])
                if out != None:
                    if i < 5:
                        edge = 'top'
                    if i > 2394:
                        edge = 'bottom'
                    return(out, edge)

        # Left/Right Col
        for i in range(1, 2399):
            for j in [0, 1, 2, 3, 4, 2999, 2998, 2997, 2996, 2995]:
                out = match_SLSTR_pixel([i, j])
                if out != None:
                    if j < 5:
                        edge = 'left'
                    if j > 2994:
                        edge = 'right'
                    return(out, edge)

        # No matches along any edge
        return(None, None)

    coords, edge = findedgepixel()

    if coords != None:
        # Check adjacent(ish) neighbours
        i = coords[0][0]
        j = coords[0][1]

        if edge == 'top':
            for i in (tqdm(range(2400)) if verbose else range(2400)):
                for k in range(j - 10, j + 10):
                    matches = match_SLSTR_pixel([i, k])
                    if matches != None:
                        coords += matches
                        j = k

        elif edge == 'bottom':
            for i in (tqdm(range(2399, -1, -1)) if verbose else range(2399, -1, -1)):
                for k in range(j - 10, j + 10):
                    matches = match_SLSTR_pixel([i, k])
                    if matches != None:
                        coords += matches
                        j = k

        elif edge == 'left':
            for j in (tqdm(range(3000)) if verbose else range(3000)):
                for k in range(i - 10, i + 11):
                    matches = match_SLSTR_pixel([k, j])
                    if matches != None:
                        coords += matches
                        i = k

        elif edge == 'right':
            for j in (tqdm(range(2999, -1, -1)) if verbose else range(2999, -1, -1)):
                for k in range(i - 10, i + 11):
                    matches = match_SLSTR_pixel([k, j])
                    if matches != None:
                        coords += matches
                        i = k
    else:
        if persistent is True:
            tqdm.write("No pixel found on edge, brute forcing")
            for i in (tqdm(range(2400)) if verbose else range(2400)):
                for j in range(3000):
                    matches = match_SLSTR_pixel([i, j])
                    if matches != None:
                        coords += matches
        else:
            tqdm.write("No pixel found on edge, skipping")
            return(None)

    # Remove duplicates
    coords = [list(x) for x in set(tuple(x) for x in coords)]

    # Sort the coordinates
    coords.sort()

    # Return position of matching coordinates in a list
    # SLSTR_row, SLSTR_column, Calipso_index

    return(coords)


if __name__ == '__main__':
    Cfilename = "D:/SatelliteData/Calipso1km/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T23-09-16ZD.hdf"
    Sfilename = "D:/SatelliteData/SLSTR/S3A_SL_1_RBT____20180401T232333_20180401T232633_20180403T041425_0179_029_301_1800_LN2_O_NT_002.SEN3"
