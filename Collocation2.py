# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:07:55 2018

@author: tomzh
"""

import requests
import xml.etree.ElementTree as ET
import CalipsoReader2 as CR
from datetime import datetime, timedelta
import numpy as np
import os
import zipfile
import io
import DataLoader as DL
from tqdm import tqdm


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

    elif r.status_code == 400:
        print('Response Code Error')
        return([])


def makeurlquery(Cfilename, timewindow=30, num=20):
    # Set download website, instrument and product type

    base = "https://scihub.copernicus.eu/s3//search?q=%20instrumentshortname:SLSTR%20AND%20%20producttype:SL_1_RBT___"

    # Load parameters from hdf file
    with CR.SDopener(Cfilename) as file:
        lat = CR.load_data(file, 'Latitude')
        lon = CR.load_data(file, 'Longitude')
        time = CR.load_data(file, 'Profile_Time')
    time += 725846400.0    # Time in UNIX
    time -= 10  # Leap second correction

    # Select which indices to use to slice list
    xs = np.linspace(0, len(lat) - 1, num + 1)
    xs = xs.astype(int)

    out = []

    for i in range(len(xs) - 1):
        query = "%20AND%20"

        # Select indices
        a = xs[i]
        b = xs[i+1]
        c = int(0.5 * (a + b))

        # Set Time query
        timestamp = datetime.utcfromtimestamp(time[c][0])
        windowstart = timestamp - timedelta(minutes=timewindow)
        windowend = timestamp + timedelta(minutes=timewindow)
        query += "beginPosition:["
        query += str(windowstart.isoformat())[:-3] + 'Z'
        query += "%20TO%20"
        query += str(windowend.isoformat())[:-3] + 'Z' + "]"

        # Set Positional query
        query += "%20AND%20(%20footprint:%22Intersects(POLYGON(("
        query += str(lon[a][0]) + "%20" + str(lat[a][0]) + str(',')
        query += str(lon[b][0]) + "%20" + str(lat[a][0]) + str(',')
        query += str(lon[b][0]) + "%20" + str(lat[b][0]) + str(',')
        query += str(lon[a][0]) + "%20" + str(lat[b][0]) + str(',')
        query += str(lon[a][0]) + "%20" + str(lat[a][0])
        query += "%20)))%22)"

        # End query with results display options
        query += "&rows=25&start=0"
        out.append(base + query)
    return(out)


def find_SLSTR_data(Cfilename, timewindow=30, num=20):
    out = []
    queries = makeurlquery(Cfilename, timewindow, num)
    Sfilenames = []
    Sdownloads = []
    print('Finding matches for ' + Cfilename)
    for query in queries:
        response = SLSTR_query(query)
        if response != []:
            out += response
            print('Match found')
    out = list(set(out))
    for i in out:
        q = i.split(',')
        Sfilenames.append(q[0])
        Sdownloads.append(q[1])
    return(Sfilenames, Sdownloads)


def match_directory(directory, timewindow=30, num=20):
    q = os.listdir(directory)
    w = [i for i in q if i[-1] == 'f']
    Data = []
    for i in range(len(w)):
        if i % 5 == 0:
            print("%s of %s files processed" % (str(i), str(len(w))))
        try:
            Sfilenames, Sdownloads = find_SLSTR_data(directory + w[i])
            if Sfilenames != []:
                with open('Matches.txt', 'a') as file:
                    for j in range(len(Sfilenames)):
                        file.write(
                            str(w[i]) + ',' + str(Sfilenames[j]) + ',' + str(Sdownloads[j]) + '\n')
                Data.append([w[i], Sfilenames, Sdownloads])
        except:
            print("Error")
            pass
    return(Data)


def ESA_download(Sdownloads, targetdirectory):
    olddir = os.getcwd()
    os.chdir(targetdirectory)
    for i in range(len(Sdownloads)):
        if i % 10 == 0:
            print("%s of %s files downloaded" % (str(i), str(len(Sdownloads))))
        Sfile = Sdownloads[i]
        print('Downloading from ' + Sfile)
        if Sfile.endswith('$value'):
            url = Sfile
        else:
            url = Sfile + '$value'
        r = requests.get(url, auth=('s3guest', 's3guest'))
        if r.status_code != 200:
            print("Error downloading " + str(Sfile))
        else:
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall()
            except:
                print("Error downloading " + str(Sfile))
    os.chdir(olddir)


def collocate(SLSTR_filename, Calipso_filename):
    # Finds pixels in both files which represent the same geographic position

    # Load SLSTR coords
    scn = DL.scene_loader(SLSTR_filename)
    scn.load(['latitude_an', 'longitude_an'])
    slat = scn['latitude_an'].values
    slon = scn['longitude_an'].values

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
        if 0 <= i < 2400 and 0 <= j < 3000:
            matches = abs(slat[i, j] - clat) < lattolerance
            if matches.any():
                loc = np.where(matches == True)
                lontolerance = (
                    lattolerance / np.cos(slat[i, j] * np.pi / 180))
                for k in loc[0]:
                    if abs(slon[i, j] - clon[k]) < lontolerance:
                        out.append([i, j, k])
        if out == []:
            out = None
        return(out)

    def findedgepixel():
        # Check near the edge of SLSTR matrix for matches

        # Top/Bottom Row
        for i in [0, 1, 2, 2399, 2398, 2397]:
            for j in range(3000):
                out = match_SLSTR_pixel([i, j])
                if out != None:
                    if i < 3:
                        edge = 'top'
                    if i > 2396:
                        edge = 'bottom'
                    return(out, edge)

        # Left/Right Col
        for i in range(1, 2399):
            for j in [0, 1, 2, 2999, 2998, 2997]:
                out = match_SLSTR_pixel([i, j])
                if out != None:
                    if j < 3:
                        edge = 'left'
                    if j > 2997:
                        edge = 'right'
                    return(out, edge)

        # No matches along any edge
        return(None, None)

    coords, edge = findedgepixel()

    if coords != None:
        print("Collocated pixel found on edge")
        # Check adjacent(ish) neighbours
        i = coords[0][0]
        j = coords[0][1]

        if edge == 'top':
            for i in tqdm(range(2400)):
                for k in range(j - 10, j + 10):
                    matches = match_SLSTR_pixel([i, k])
                    if matches != None:
                        coords += matches
                        j = k

        elif edge == 'bottom':
            for i in tqdm(range(2399, -1, -1)):
                for k in range(j - 10, j + 10):
                    matches = match_SLSTR_pixel([i, k])
                    if matches != None:
                        coords += matches
                        j = k

        elif edge == 'left':
            for j in tqdm(range(3000)):
                for k in range(i - 10, i + 11):
                    matches = match_SLSTR_pixel([k, j])
                    if matches != None:
                        coords += matches
                        i = k

        elif edge == 'right':
            for j in tqdm(range(2999, -1, -1)):
                for k in range(i - 10, i + 11):
                    matches = match_SLSTR_pixel([k, j])
                    if matches != None:
                        coords += matches
                        i = k
    else:
        print("No pixel found on edge, skipping")
        return(None)

    # Remove duplicates
    coords = [list(x) for x in set(tuple(x) for x in coords)]

    # Sort the coordinates
    coords.sort()

    # Return position of matching coordinates in a list
    # SLSTR_row, SLSTR_column, Calipso_index
    return(coords)


if __name__ == '__main__':
    Cfilename = "D:/SatelliteData/Calipso1km/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T00-04-48ZD.hdf"
    Sfilename = "D:/SatelliteData/S3A_SL_1_RBT____20180401T012743_20180401T013043_20180402T055007_0179_029_288_1620_LN2_O_NT_002.SEN3"
