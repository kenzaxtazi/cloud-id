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
    r = requests.get(url, auth=('s3guest', 's3guest'))
    root = ET.fromstring(r.text)
    out = []
    # Number of matches:
    if 'totalResult' in str(root[5]):
        num_matches = root[5].text
    else:
        num_matches = 0
        
    if num_matches == 0:
        print('No matches')
        
    else:
        for i in root:
            if "entry" in str(i):
                out.append(i[0].text + "," + i[1].attrib['href'])
    return(out)
    
def makeurlquery(Cfilename, timewindow=30, num=20):
    # Set download website, instrument and product type

    base = "https://scihub.copernicus.eu/s3//search?q=%20instrumentshortname:SLSTR%20AND%20%20producttype:SL_1_RBT___"

    
    # Load parameters from hdf file
    file = CR.load_hdf(Cfilename)
    lat = CR.load_data(file, 'Latitude')
    lon = CR.load_data(file, 'Longitude')
    time = CR.load_data(file, 'Profile_Time')   # Time in IAT
    time += 725846400.0    # Time in UNIX 
    time -= 10 # Leap second correction
    
    # Select which indices to use to slice list
    xs = np.linspace(0, len(lat) - 1, num + 1)
    xs = xs.astype(int)
    
    out = []

    # Set time query
    for j in range(2):
        longquery = "%20AND%20%20"
        for i in range(int(num/2)):
            query = "("
            a = xs[i]
            b = xs[i+1]
            c = int(0.5 * (a + b))
            timestamp = datetime.utcfromtimestamp(time[c][0])
            windowstart = timestamp - timedelta(minutes = timewindow)
            windowend = timestamp + timedelta(minutes = timewindow)
            query += "beginPosition:["
            query += str(windowstart.isoformat())[:-3] + 'Z'
            query += "%20TO%20"
            query += str(windowend.isoformat())[:-3] + 'Z' + "]"
    
        
        # Set Positional query
            query += "%20%20%20AND%20%20(%20footprint:%22Intersects(POLYGON(("
            query += str(lon[a][0]) + "%20" + str(lat[a][0]) + str(',')
            query += str(lon[b][0]) + "%20" + str(lat[a][0]) + str(',')
            query += str(lon[b][0]) + "%20" + str(lat[b][0]) + str(',')
            query += str(lon[a][0]) + "%20" + str(lat[b][0]) + str(',')
            query += str(lon[a][0]) + "%20" + str(lat[a][0])
            query += "%20)))%22))"
            
            if i == 0:
                longquery += query
            else:
                longquery += "%20OR%20" + query
        
        longquery += "&rows=25&start=0"
        out.append(base + longquery)
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
            print("%s of %s files processed"%(str(i), str(len(w))))
        try:
            Sfilenames, Sdownloads = find_SLSTR_data(directory + w[i])
            if Sfilenames != []:
                with open('Matches.txt', 'a') as file:
                    for j in range(len(Sfilenames)):
                        file.write(str(w[i]) + ',' + str(Sfilenames[j]) + ',' + str(Sdownloads[j]) + '\n')
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
            print("%s of %s files downloaded"%(str(i), str(len(Sdownloads))))
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
    file = CR.load_hdf(Calipso_filename)
    clat = CR.load_data(file, 'Latitude')
    clon = CR.load_data(file, 'Longitude')
    
    # Find coord pairs which are close
    coords = []
    
    # Want the latitude and longitude to be within 250m of each others
    # 250m = 0.00224577793 degrees lon at equator
    # 250m = 0.00224577793 * cos(lat) degrees lon at lat
#    lontolerance = 0.00224577793 * np.cos(clat[0] * np.pi / 180)
    lattolerance = 0.00224577793
    #check edges
    
    # Edge flag
    top, bottom, left, right = False, False, False, False
    
    # Top/Bottom Row
    for i in [0, 1, 2, 2397, 2398, 2399]:
        for j in range(3000):
            matches = abs(slat[i, j] - clat) < lattolerance
            if matches.any():
                loc = np.where(matches == True)
                lontolerance = 0.00224577793 / np.cos(slat[i, j] * np.pi / 180)
                for k in loc[0]:
                    if abs(slon[i, j] - clon[k]) < lontolerance:
                        if i == 0:
                            top = True
                        if i == 2399:
                            bottom = True
                        coords.append([i, j, k])
            
    # Left/Right Col
    for i in range(1, 2399):
        for j in [0, 1, 2, 2997, 2998, 2999]:
            matches = abs(slat[i, j] - clat) < lattolerance
            if matches.any():
                loc = np.where(matches == True)
                lontolerance = 0.00224577793 / np.cos(slat[i, j] * np.pi / 180)
                for k in loc[0]:
                    if abs(slon[i, j] - clon[k]) < lontolerance:
                        if j == 0:
                            left = True
                        if j == 2999:
                            right = True
                        coords.append([i, j, k])
            
    if len(coords) != 0:
        print("Collocated pixel found on edge")
        # Check adjacent(ish) neighbours
        i = coords[0][0]
        j = coords[0][1]

        
        if top == True:
            for i in tqdm(range(2400)):
                for k in range(j - 10, j + 10):
                    try:
                        matches = abs(slat[i, k] - clat) < lattolerance
                        if matches.any():
                            loc = np.where(matches == True)
                            lontolerance = 0.00224577793 / np.cos(slat[i, k] * np.pi / 180)
                            for l in loc[0]:
                                if abs(slon[i, k] - clon[l]) < lontolerance:
                                    coords.append([i, k, l])
                                    j = k
                    except IndexError:
                        pass
                    
        elif bottom == True:
            for i in tqdm(range(2399, -1, -1)): 
                for k in range(j - 10, j + 10):
                    try:
                        matches = abs(slat[i, k] - clat) < lattolerance
                        if matches.any():
                            loc = np.where(matches == True)
                            lontolerance = 0.00224577793 / np.cos(slat[i, k] * np.pi / 180)
                            for l in loc[0]:
                                if abs(slon[i, k] - clon[l]) < lontolerance:
                                    coords.append([i, k, l])
                                    j = k
                    except IndexError:
                        pass
                
        elif left == True:
            for j in tqdm(range(3000)):
                for k in range(i - 10, i + 11):
                    try:
                        matches = abs(slat[k, j] - clat) < lattolerance
                        if matches.any():
                            loc = np.where(matches == True)
                            lontolerance = 0.00224577793 / np.cos(slat[k, j] * np.pi / 180)
                            for l in loc[0]:
                                if abs(slon[k, j] - clon[l]) < lontolerance:
                                    coords.append([k, j, l])
                                    i = k
                    except IndexError:
                        pass
                    
        elif right == True:
            for j in tqdm(range(2999, -1, -1)):
                for k in range(i - 10, i + 11):
                    try:
                        matches = abs(slat[k, j] - clat) < lattolerance
                        if matches.any():
                            loc = np.where(matches == True)
                            lontolerance = 0.00224577793 / np.cos(slat[k, j] * np.pi / 180)
                            for l in loc[0]:
                                if abs(slon[k, j] - clon[l]) < lontolerance:
                                    coords.append([k, j, l])
                                    i = k
                    except IndexError:
                        pass
                                
    else:
        print("No pixel found on edge")
        return(None)
    coords = [list(x) for x in set(tuple(x) for x in coords)]
    # Return position of matching coordinates in a row
    # SLSTR_row, SLSTR_column, Calipso_index
    return(coords)
    
if __name__ == '__main__':
    Cfilename = "D:/SatelliteData/Calipso1km/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T00-04-48ZD.hdf"
    Sfilename = "D:/SatelliteData/S3A_SL_1_RBT____20180401T012743_20180401T013043_20180402T055007_0179_029_288_1620_LN2_O_NT_002.SEN3"
    