# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:12:47 2018

@author: tomzh
"""

import subprocess
import DataLoader as DL
import CalipsoReader2 as CR
import platform
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm

# Calipso File
Cfilename = "D:/SatelliteData/Calipso1km/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T00-04-48ZD.hdf"
Sfilename = "D:/SatelliteData/S3A_SL_1_RBT____20180401T012743_20180401T013043_20180402T055007_0179_029_288_1620_LN2_O_NT_002.SEN3"

def find_SLSTR_data(filename, timewindow=30, num=20, dryrun=False, outputdir=None, silent=True):
    data = []
    
    # Set download website, user credentials, instruments and product type
    command = ["-d", "https://scihub.copernicus.eu/s3/", "-u", "s3guest", "-p", "s3guest", "-i", "SLSTR", "-T", "SL_1_RBT___"]
    
    # Set correct commands for executing the .sh file. System dependent.
    if platform.platform()[:10] == 'Windows-10': #
        command = ["C:/Program Files/Git/git-bash.exe", "dhusget.sh"] + command
    else:
        command = ["./dhusget.sh"] + command
    
    if outputdir != None:
        command = command + ["-o", "'product'", "-O", str(outputdir)]
    # Load parameters from hdf file
    file = CR.load_hdf(filename)
    lat = CR.load_data(file, 'Latitude')
    lon = CR.load_data(file, 'Longitude')
    time = CR.load_data(file, 'Profile_Time')   # Time in IAT
    time += 725846400.0    # Time in UNIX 
    time -= 10 # Leap second correction
    
    # Select which indices to use to slice list
    xs = np.linspace(0, len(lat) - 1, num + 1)
    xs = xs.astype(int)
    
    
    # Set time query
    for i in range(len(xs) - 1):
        query = []
        a = xs[i]
        b = xs[i+1]
        c = int(0.5 * (a + b))
        timestamp = datetime.utcfromtimestamp(time[c][0])
        windowstart = timestamp - timedelta(minutes = timewindow)
        windowend = timestamp + timedelta(minutes = timewindow)
        query = query + ["-S", str(windowstart.isoformat())[:-3] + 'Z']
        query = query + ["-E", str(windowend.isoformat())[:-3] + 'Z']
    
        # Set Positional query
        query = query + ["-c", str(lon[a][0]) + ',' + str(lat[a][0]) + ':' + str(lon[b][0]) + ',' + str(lat[b][0])]
        
        # Send query
        if dryrun == False:
        
            subprocess.call(command + query)
            with open("products-list.csv", "r") as file:
                data += file.readlines()
                

        # For now it puts any matching SLSTR file into a XML and CSV then
        # Overwrites these files on the next loop
        
        print(query)
    return(data)

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
    for i in [0, 2399]:
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
        for j in [0, 2999]:
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
                    matches = abs(slat[i, k] - clat) < lattolerance
                    if matches.any():
                        loc = np.where(matches == True)
                        lontolerance = 0.00224577793 / np.cos(slat[i, k] * np.pi / 180)
                        for l in loc[0]:
                            if abs(slon[i, k] - clon[l]) < lontolerance:
                                coords.append([i, k, l])
                                j = k
        elif bottom == True:
            for i in tqdm(range(2399, -1, -1)):
                for k in range(j - 10, j + 10):
                    matches = abs(slat[i, k] - clat) < lattolerance
                    if matches.any():
                        loc = np.where(matches == True)
                        lontolerance = 0.00224577793 / np.cos(slat[i, k] * np.pi / 180)
                        for l in loc[0]:
                            if abs(slon[i, k] - clon[l]) < lontolerance:
                                coords.append([i, k, l])
                                j = k             
                
        elif left == True:
            for j in tqdm(range(3000)):
                for k in range(i - 10, i + 11):
                    matches = abs(slat[k, j] - clat) < lattolerance
                    if matches.any():
                        loc = np.where(matches == True)
                        lontolerance = 0.00224577793 / np.cos(slat[k, j] * np.pi / 180)
                        for l in loc[0]:
                            if abs(slon[k, j] - clon[l]) < lontolerance:
                                coords.append([k, j, l])
                                i = k

        elif right == True:
            for j in tqdm(range(2999, -1, -1)):
#            while j > 0:
#                print(j)
#                j -= 1
                for k in range(i - 10, i + 11):
                    matches = abs(slat[k, j] - clat) < lattolerance
                    if matches.any():
                        loc = np.where(matches == True)
                        lontolerance = 0.00224577793 / np.cos(slat[k, j] * np.pi / 180)
                        for l in loc[0]:
                            if abs(slon[k, j] - clon[l]) < lontolerance:
                                coords.append([k, j, l])
                                i = k
                                
    else:
        print("No pixel found on edge")
        for i in tqdm(range(2400)):
            for j in range(3000):
                matches = abs(slat[i, j] - clat) < lattolerance
                if matches.any():
                    loc = np.where(matches == True)
                    lontolerance = 0.00224577793 / np.cos(slat[i, j] * np.pi / 180)
                    for k in loc[0]:
                        if abs(slon[i, j] - clon[k]) < lontolerance:
                            coords.append([i, j, k])
    
    # Return position of matching coordinates in a row
    # SLSTR_row, SLSTR_column, Calipso_index
    return(coords)
    

