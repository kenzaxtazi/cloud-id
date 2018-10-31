# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:46:43 2018

@author: tomzh
"""
import re
import os
from satpy import Scene
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from time import time
from ftplib import FTP
from getpass import getuser


def FTPlogin():
    ftp = FTP('ftp.ceda.ac.uk')
    with open('credentials.txt', 'r') as file:
        username, password = file.readlines()
        ftp.login(username.strip(), password.strip())
    ftp.cwd('neodc/sentinel3a/data/SLSTR/L1_RBT')
    return(ftp)

def FTPdownload(ftpobj, path, destination):
    startdir = os.getcwd()
    os.chdir(destination)
    if path[:3] == "S3A":   # given path is folder name
        foldername = path
        path = path = path[16:20] + '/' + path[20:22] + '/' + path[22:24] + '/' + path[:]
    elif path[:2] == "20":   # given path is path from /L1_RBT
        foldername = path[11:]
    try:
        ftpobj.retrbinary("RETR " + str(path), open(str(foldername), "wb").write)
    except PermissionError:
        print("Permission Error")
        print(foldername)
    os.chdir(startdir)
    print('Download complete')
    
 
def fixdir(list_in):
    for i in range(len(list_in)):
        list_in[i] = list_in[i].replace('\\', '/')
    return(list_in)
    
def _regpattern():
    cpattern = re.compile("(?P<mission_id>.{3})\_SL\_(?P<processing_level>.{1})\_(?P<datatype_id>.{6})\_(?P<start_time>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})\_(?P<end_time>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})\_(?P<creation_time>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})\_(?P<duration>.{4})\_(?P<cycle>.{3})\_(?P<relative_orbit>.{3})\_(?P<frame>.{4})\_(?P<centre>.{3})\_(?P<mode>.{1})\_(?P<timeliness>.{2})\_(?P<collection>.{3})\.zip")
    return(cpattern)

def _regpatternf():
    fpattern = re.compile("(?P<mission_id>.{3})\_SL\_(?P<processing_level>.{1})\_(?P<datatype_id>.{6})\_(?P<start_time>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})\_(?P<end_time>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})\_(?P<creation_time>\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})\_(?P<duration>.{4})\_(?P<cycle>.{3})\_(?P<relative_orbit>.{3})\_(?P<frame>.{4})\_(?P<centre>.{3})\_(?P<mode>.{1})\_(?P<timeliness>.{2})\_(?P<collection>.{3})\.SEN3")
    return(fpattern)


def foldermatch_dict(folder):
    cpattern = _regpattern()
    out = [m.groupdict() for m in cpattern.finditer(folder)]
    return(out)
   

def foldermatch(folder):
    cpattern = _regpattern()
    out = re.findall(cpattern, folder)
    return(out)    


def find_files(ftp):
    filenames = open("filenames18.txt", "w")
    years = ["2016", "2017", "2018"]
    months = ["%.2d" % i for i in range(1, 13)]
    days = ["%.2d" % i for i in range(1, 32)]
    for year in years[2:]:
        ftp.cwd(year)
        print(year)
        for month in months:
            print("m", month)
            try:
                ftp.cwd(month)
                for day in days:
                    print(day)
                    try:
                        ftp.cwd(day)            
                        files_in_dir = ftp.nlst()
                        for folder in files_in_dir:
                            if foldermatch(folder) != []:
                                print(folder)
                                filenames.write(str(folder))
                                filenames.write("\n")
                        ftp.cwd("..")
                    except:
                        pass
                    
                ftp.cwd("..")
            except:
                pass
        ftp.cwd("..")
    filenames.close()

def find_files_for_pos(rel_orbit, frame, centre=None):
    filefolders = ["filenames16.txt", "filenames17.txt", "filenames18.txt"]
    out = []
    for filefolder in filefolders:
        with open(filefolder, 'r') as filenamefolder:
            names = filenamefolder.readlines()
            for name in names:
                regexlist = foldermatch(name)
                file_rel_orbit = int(regexlist[0][8])
                if file_rel_orbit == rel_orbit:
                    file_frame = int(regexlist[0][9])
                    if abs(file_frame - frame) < 1:
                        if centre == None:
                            out.append(name.strip())
                        elif centre == regexlist[0][10]:
                            out.append(name.strip())
                        else:
                            pass
    out.sort()
    return(out)




def path_to_public():
    user = getuser()
    path = ("/home/hep/" + str(user) + "/public_html")
    return(path)
    
    
    
def scene_loader(path):
    # Returns a satpy scene object from the provided file
    if path[-1] == '/':
        path = path + "*"
    elif path[-1] == '*':
        pass
    else:
        path = path + "/*"
    filenames = glob(path)
    filenames = fixdir(filenames)
    scn = Scene(filenames=filenames, reader='nc_slstr')
    return(scn)

def mask_analysis(scn):
    # Loads the a masks from simplistic and bayesian files. 
    # WIP: Creates figure of all simplistic masks
    scn.load(['cloud_an', 'bayes_an'])
    """ 
    Cloud_bn file
    Flag masks: Flag meanings
    1: Visible 1.37_threshold
    2: 1.6_small_histogram
    4: 1.6_large_histogram
    8: 2.25_small_histogram
    16: 2.25 large_histogram
    32: 11_spatial_coherence
    64: gross_cloud
    128: thin_cirrus
    256: medium_high
    512: fog_low_stratus
    1024: 11_12_view_difference
    2048: 3.7.11_view_difference
    4096: thermal_histogram
    8192: spare
    16384: spare
    
    Bayes_bn file
    Flag masks: Flag meanings
    1: single_low
    2: single_moderate
    4: dual_low
    8: dual_moderate
    16: spare
    32: spare
    64: spare
    128: spare
    """
    for mask in scn['cloud_bn'].flag_masks[:-2]:
        plt.figure()
        plt.title(str(mask))
        data = scn['cloud_bn'].values & mask
        plt.imshow(data)
        
    
    
def summary(scene, filenames=None, saveimage=False, outputpath='public'):
    # Loads positional S1_n channel data. Prints lat/lon of corner pixel
    # If saveimage is True, saves png to current directory with metadata
    scene.load(['S1_n', 'latitude', 'longitude'])
    lat = scene['latitude'].values[0][0] # Latitude of corner pixel
    lon = scene['longitude'].values[0][0] # Longitude of corner pixel
    if saveimage != False:
        if outputpath == 'public':
            # cd to public folder
            os.chdir(path_to_public())
        if filenames != None:
            imagename = ('S1n_' + str(filenames[0][:31]) + '_' + 
                         str(filenames[0][82:94]) + '-(' + str(lat) + ',' + 
                         str (lon) +')')
        else:
            imagename = 'test'
        scene.save_dataset('S1_n', str(imagename) + '.png')
    print(str(lat) + ', ' + str(lon))


def makepltimage(scene, channel='S1_n'):
    # Use matplotlib to produce image of specified channel
    scene.load([channel])
    data = scene[channel].values
    data = np.nan_to_num(data)
    plt.figure()
    plt.imshow(data)
    
    
def makepngimage(scene, channel='S1_n', outputpath='public'):
    if outputpath == 'public':
        # cd to public folder
        os.chdir(path_to_public())
    scene.save_dataset(channel, str(time()) + '.png')
    
    