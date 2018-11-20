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
        Sfilenames, Sdownloads = find_SLSTR_data(directory + w[i])
        if Sfilenames != []:
            with open('Matches.txt', 'a') as file:
                for j in range(len(Sfilenames)):
                    file.write(str(w[i]) + ',' + str(Sfilenames[j]) + ',' + str(Sdownloads[j]) + '\n')
            Data.append([w[i], Sfilenames, Sdownloads])
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
            
            
if __name__ == '__main__':
    #url = "https://scihub.copernicus.eu/s3//search?q=%20instrumentshortname:SLSTR%20AND%20producttype:SL_1_RBT___%20AND%20(%20footprint:%22Intersects(POLYGON((101.1878500000000%2016.7101170000000,98.9603900000000%2016.7101170000000,98.9603900000000%2026.2178780000000,101.1878500000000%2026.2178780000000,101.1878500000000%2016.7101170000000%20)))%22)&rows=25&start=0"
    #print(SLSTR_query(url))
    pass
    