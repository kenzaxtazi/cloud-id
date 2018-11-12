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
filename = "D:/SatelliteData/Calipso1km/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T00-04-48ZD.hdf"


def find_SLSTR_data(filename, timewindow=30, num=10):
    # Set download website, user credentials and instrument
    command = ["-d", "https://scihub.copernicus.eu/s3/", "-u", "s3guest", "-p", "s3guest", "-i", "SLSTR"]
    
    # Set correct commands for executing the .sh file. System dependent.
    if platform.platform()[:10] == 'Windows-10': #
        command = ["C:/Program Files/Git/git-bash.exe", "dhusget.sh"] + command
    else:
        command = ["./dhusget.sh"] + command
    
    # Load parameters from hdf file
    file = CR.load_hdf(filename)
    lat = CR.load_data(file, 'Latitude')
    lon = CR.load_data(file, 'Longitude')
    time = CR.load_data(file, 'Profile_Time')   # Time in IAT
    time += 725846400.0    # Time in UNIX 
    
    # Select which indices to use to slice list
    xs = np.linspace(0, len(lat) - 1, num + 1)
    xs = xs.astype(int)
    
    
    # Set time query
    for i in range(len(xs) - 1):
        query = []
        a = xs[i]
        b = xs[i+1]
        c = int(0.5 * (a + b))
        timestamp = datetime.fromtimestamp(time[c][0])  # Inaccurate by a few leap seconds
        windowstart = timestamp - timedelta(minutes = timewindow)
        windowend = timestamp + timedelta(minutes = timewindow)
        query = query + ["-S", str(windowstart.isoformat())[:-3] + 'Z']
        query = query + ["-E", str(windowend.isoformat())[:-3] + 'Z']
    
        # Set Positional query
        query = query + ["-c", str(lon[a][0]) + ',' + str(lat[a][0]) + ':' + str(lon[b][0]) + ',' + str(lat[b][0])]
        
        # Send query
        
        # subprocess.call(command + query)

        # For now it puts any matching SLSTR file into a XML and CSV then
        # Overwrites these files on the next loop
        
        print(query)

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
    # Want the latitude / longitude to be within 250m of each others
    # 250m = 0.00224577793 degrees lon at equator
    # 250m = 0.00224577793 * cos(lat) degrees lon at lat
    lontolerance = 0.0224577793 * np.cos(clat[0] * np.pi / 180)
    lattolerance = 0.0224577793
    #check edges
    
    # Edge flag
    top, bottom, left, right = False, False, False, False
    
    # Top/Bottom Row
    for i in [0, 2399]:
        for j in range(3000):
            matches = np.isclose(slat[i, j], clat, rtol=lattolerance)
            if matches.any():
                loc = np.where(matches == True)
                if np.isclose(slon[i, j], clon[loc[0][0]], rtol=lontolerance):
                    coords.append([i, j, loc[0][0]])
                    if i == 0:
                        top = True
                    elif i == 2399:
                        bottom = True
    
    # Left/Right Col
    for i in range(1, 2399):
        for j in [0, 2999]:
            matches = np.isclose(slat[i, j], clat, rtol=lattolerance)
            if matches.any():
                loc = np.where(matches == True)
                if np.isclose(slon[i, j], clon[loc[0][0]], rtol=lontolerance):
                    coords.append([i, j, loc[0][0]])
                    if j == 0:
                        left = True
                    if j == 2999:
                        right = True
            
    if len(coords) == 0:
        print('No matches found')
        return(coords)
    
    edgeflagsum = top + right + left + bottom
    if edgeflagsum == 2:
        # Use interpolation between two edges to find which pixels to check
        xedge = [i[0] for i in coords]
        yedge = [i[1] for i in coords]
        xs = np.linspace(np.min(xedge) + 1, np.max(xedge) - 1, np.max(xedge) - np.min(xedge) - 1).astype(int)
        ys = np.interp(xs, xedge, yedge)
        ys = ys.astype(int)

        
        for i in range(len(xs)):
            ylist = [ys[i]]
            if ys[i] != 0:
                ylist.append(i-1)
            if ys[i] != 2999:
                ylist.append(i+1)
            for y in ylist:
                matches = np.isclose(slat[xs[i], y], clat, rtol=lattolerance)
                if matches.any():
                    loc = np.where(matches == True)
                    if np.isclose(slon[xs[i], y], clon[loc[0][0]], rtol=lontolerance):
                        coords.append([xs[i], y, loc[0][0]])
        
        return(coords, xedge, yedge)
        
    elif edgeflagsum > 2:
        # Should never happen 
        print('Warning: More than 2 Edges detected.')
        
    elif edgeflagsum < 2:
        # Occurs if a region is NaN or the Calipso Granule starts/ends in frame
        print('Warning: Less than 2 Edges detected.')
        
    
    for i in tqdm(range(2400)):
        for j in range(3000):
            if (np.isclose(slat[i, j], clat, rtol=1e-3)).any():
                loc = np.where(np.isclose(slat[i, j], clat, rtol=1e-3) == True)
                if np.isclose(slon[i, j], clon[loc[0][0]], rtol=1e-3):
                    coords.append([i, j, loc[0][0]])
    
    # Return position of matching coordinates in a row
    # SLSTR_row, SLSTR_column, Calipso_index
    return(coords)