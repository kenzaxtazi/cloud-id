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
    
    # Find coord pairs which are close, - latitude dependence needs to be added
    coords = []
    for i in tqdm(range(2400)):
        for j in range(3000):
            if (np.isclose(slat[i, j], clat)).any():
                loc = np.where(np.isclose(slat[i, j], clat) == True)
                if np.isclose(slon[i, j], clon[loc[0][0]]):
                    coords.append([i, j, loc[0][0]])
    
    # Return position of matching coordinates in a row
    # SLSTR_row, SLSTR_column, Calipso_index
    return(coords)