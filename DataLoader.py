# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:46:43 2018

@author: tomzh
"""
import os
import platform
import re
import zipfile
from ftplib import FTP
from getpass import getuser
from glob import glob
from time import time
from pyhdf.SD import SD, SDC

import matplotlib.pyplot as plt
import numpy as np
from satpy import Scene


def FTPlogin(creds_path='credentials.txt'):
    ftp = FTP('ftp.ceda.ac.uk')
    with open(creds_path, 'r') as file:
        username, password = file.readlines()
        ftp.login(username.strip(), password.strip())
    ftp.cwd('neodc/sentinel3a/data/SLSTR/L1_RBT')
    return(ftp)


def FTPdownload(ftpobj, path, destination):
    startdir = os.getcwd()
    os.chdir(destination)
    if path[:3] == "S3A":   # given path is folder name
        foldername = path
        path = path = path[16:20] + '/' + path[20:22] + \
            '/' + path[22:24] + '/' + path[:]
    elif path[:2] == "20":   # given path is path from /L1_RBT
        foldername = path[11:]
    try:
        ftpobj.retrbinary("RETR " + str(path),
                          open(str(foldername), "wb").write)
    except:
        print("Permission Error")
        print(foldername)
        try:
            os.remove(foldername)
        except:
            pass
    os.chdir(startdir)
    print('Download complete')


def fixdir(list_in):
    for i in range(len(list_in)):
        list_in[i] = list_in[i].replace('\\', '/')
    return(list_in)


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

    olddir = os.getcwd()

    if platform.platform()[:10] == "Windows-10":
        string1 = "S3A_SL_1"
        index = path.find(string1)
        if index == 0:
            pass
        else:
            newdir = path[:index]
            os.chdir(newdir)
            path = path[index:]
    filenames = glob(path)
    filenames = fixdir(filenames)
    scn = Scene(filenames=filenames, reader='slstr_l1b')
    os.chdir(olddir)
    return(scn)


def summary(scene, filenames=None, saveimage=False, outputpath='public'):
    # Loads positional S1_n channel data. Prints lat/lon of corner pixel
    # If saveimage is True, saves png to current directory with metadata
    scene.load(['S1_an', 'latitude', 'longitude'])
    lat = scene['latitude'].values[0][0]  # Latitude of corner pixel
    lon = scene['longitude'].values[0][0]  # Longitude of corner pixel
    if saveimage != False:
        if outputpath == 'public':
            # cd to public folder
            os.chdir(path_to_public())
        if filenames != None:
            imagename = ('S1n_' + str(filenames[0][:31]) + '_' +
                         str(filenames[0][82:94]) + '-(' + str(lat) + ',' +
                         str(lon) + ')')
        else:
            imagename = 'test'
        scene.save_dataset('S1_an', str(imagename) + '.png')
    print(str(lat) + ', ' + str(lon))


def makepltimage(scene, channel='S1_an'):
    # Use matplotlib to produce image of specified channel
    scene.load([channel])
    data = scene[channel].values
    data = np.nan_to_num(data)
    plt.figure()
    plt.imshow(data, cmap='gray')


def makepngimage(scene, channel='S1_an', outputpath='public'):
    if outputpath == 'public':
        # cd to public folder
        os.chdir(path_to_public())
    scene.save_dataset(channel, str(time()) + '.png')


def create_mask(scn, mask_name):
    """Extracts bitmasks and combines them into an overall mask array"""
    mask = []
    for bitmask in scn[mask_name].flag_masks[:-2]:
        data = scn[mask_name].values & bitmask
        mask.append(data)
    mask = np.sum(mask, axis=0)
    return mask


def norm(band):
    """ Normalises the bands for the false color image"""
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))


def false_color_image(band1, band2, band3, plot=True):
    """ 
    Creates a false colour image

    Input: 
    band1 (2D array) <=> red 
    band2 (2D array) <=> green
    band3 (2D array) <=> blue

    Output: 6D array (3*2D)

    if: plot= True, the image is plotted
    """
    rgb = np.dstack((norm(band1), norm(band2), norm(band3)))

    if plot == True:
        plt.figure()
        plt.imshow(rgb)
        plt.title('False colour image')

    return rgb


def mask(mask, mask_name, background):
    """Plots a semi-transparent mask over a background image"""
    plt.figure()
    plt.imshow(background, 'gray')
    plt.title(mask_name)
    mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    mask = np.ma.masked_where(mask < 1, mask)
    #X, Y = np.meshgrid(np.arange(0,3000),np.arange(0, 2400))
    #plt.contour(X, Y, mask, levels=[0., 1.0], cmap='inferno', alpha=0.3)
    plt.imshow(mask, vmin=0, vmax=1.1, cmap='inferno', alpha=0.3)


def load_hdf(filename):
    """Loads the hdf4 object into memory"""
    file = SD(filename, SDC.READ)
    return(file)


def get_header_names(file):
    """Print the names of the dataset names"""
    datasets_dic = file.datasets()
    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)


def load_data(file, variable):
    """From the file, load the chosen variable. Valid options in get_header_names()"""
    sds_obj = file.select(variable)
    data = sds_obj.get()
    return(data)


class SDopener():
    # Class to call when using context manager
    def __init__(self, path, mode=SDC.READ):
        self.path = path
        self.mode = mode

    def __enter__(self):
        self.SD = SD(self.path, self.mode)
        return(self.SD)

    def __exit__(self, type, value, traceback):
        self.SD.__del__()


def vfm_feature_flags(val):
    """ Python version of the IDL code to read the bitwise flags"""

    feature_type = val & 7
    # feature_type_qa = (val >> 3) & 3
    # ice_water_phase = (val >> 5) & 3
    # ice_water_phase_qa = (val >> 7) & 3
    # feature_subtype = (val >> 9) & 7
    # cloud_aerosol_psc_type_qa = (val >> 12) & 1
    # horizontal_averaging = (val >> 13) & 7

    # if feature_type == 0:
    #     print("Feature Type : invalid (bad or missing data)")

    # if feature_type == 1:
    #     print("Feature Type : clean air")

    # if feature_type == 2:
    #     print("Feature Type : cloud")
    #     if feature_subtype == 0:
    #         print("Feature Subtype : low overcast, transparent")
    #     if feature_subtype == 1:
    #         print("Feature Subtype : low overcast, opaque")
    #     if feature_subtype == 2:
    #         print("Feature Subtype : transition stratocumulus")
    #     if feature_subtype == 3:
    #         print("Feature Subtype : low, broken cumulus")
    #     if feature_subtype == 4:
    #         print("Feature Subtype : altocumulus (transparent)")
    #     if feature_subtype == 5:
    #         print("Feature Subtype : altostratus (opaque)")
    #     if feature_subtype == 6:
    #         print("Feature Subtype : cirrus (transparent)")
    #     if feature_subtype == 7:
    #         print("Feature Subtype : deep convective (opaque)")
    #     else :
    #         print("*** error getting Feature Subtype")

    # if feature_type == 3:
    #     print("Feature Type : aerosol")
    #     if feature_subtype == 0:
    #         print("Feature Subtype : not determined")
    #     if feature_subtype == 1:
    #         print("Feature Subtype : clean marine")
    #     if feature_subtype == 2:
    #         print("Feature Subtype : dust")
    #     if feature_subtype == 3:
    #         print("Feature Subtype : polluted continental")
    #     if feature_subtype == 4:
    #         print("Feature Subtype : clean continental")
    #     if feature_subtype == 5:
    #         print("Feature Subtype : polluted dust")
    #     if feature_subtype == 6:
    #         print("Feature Subtype : smoke")
    #     if feature_subtype == 7:
    #         print("Feature Subtype : other")
    #     else :
    #         print("*** error getting Feature Subtype")

    # if feature_type == 4:
    #     print("Feature Type : stratospheric feature--PSC or stratospheric aerosol")
    #     if feature_subtype == 0:
    #         print("Feature Subtype : not determined")
    #     if feature_subtype == 1:
    #         print("Feature Subtype : non-depolarizing PSC")
    #     if feature_subtype == 2:
    #         print ("Feature Subtype : depolarizing PSC")
    #     if feature_subtype == 3:
    #         print, "Feature Subtype : non-depolarizing aerosol"
    #     if feature_subtype == 4:
    #         print("Feature Subtype : depolarizing aerosol")
    #     if feature_subtype == 5:
    #         print("Feature Subtype : spare")
    #     if feature_subtype == 6:
    #         print("Feature Subtype : spare")
    #     if feature_subtype == 7:
    #         print("Feature Subtype : other")
    #     else:
    #         print("*** error getting Feature Subtype")

    # if feature_type == 5:
    #     print("Feature Type : surface")
    # if feature_type == 6:
    #     print("Feature Type : subsurface")
    # if feature_type == 7:
    #     print("Feature Type : no signal (totally attenuated)")
    # else :
    #     print("*** error getting Feature Type")

    # if feature_type_qa == 0:
    #     print("Feature Type QA : none")
    # if feature_type_qa == 1:
    #     print("Feature Type QA : low")
    # if feature_type_qa == 2:
    #     print("Feature Type QA : medium")
    # if feature_type_qa == 3:
    #     print("Feature Type QA : high")
    # else :
    #     print("*** error getting Feature Type QA")

    # if ice_water_phase == 0:
    #     print("Ice/Water Phase : unknown/not determined")
    # if ice_water_phase == 1:
    #     print("Ice/Water Phase : ice")
    # if ice_water_phase == 2:
    #     print("Ice/Water Phase : water")
    # if ice_water_phase == 3:
    #     print("Ice/Water Phase : mixed phase")
    # else :
    #     print("*** error getting Ice/Water Phase")

    # if ice_water_phase_qa == 0:
    #     print("Ice/Water Phase QA: none")
    # if ice_water_phase_qa == 1:
    #     print("Ice/Water Phase QA: low")
    # if ice_water_phase_qa == 2:
    #     print("Ice/Water Phase QA: medium")
    # if ice_water_phase_qa == 3:
    #     print("Ice/Water Phase QA: high")
    # else :
    #     print("*** error getting Ice/Water Phase QA")

    # if cloud_aerosol_psc_type_qa == 0:
    #     print("Cloud/Aerosol/PSC Type QA : not confident")
    # else:
    #     print("Cloud/Aerosol/PSC Type QA : confident")

    # if horizontal_averaging == 0:
    #     print("Horizontal averaging required for detection: not applicable")
    # if horizontal_averaging == 1:
    #     print("Horizontal averaging required for detection: 1/3 km")
    # if horizontal_averaging == 2:
    #     print("Horizontal averaging required for detection: 1 km")
    # if horizontal_averaging == 3:
    #     print("Horizontal averaging required for detection: 5 km")
    # if horizontal_averaging == 4:
    #     print("Horizontal averaging required for detection: 20 km")
    # if horizontal_averaging == 5:
    #     print("Horizontal averaging required for detection: 80 km")
    # else :
    #     print("*** error getting Horizontal averaging")
    return(feature_type)


if __name__ == '__main__':

    # create readers and open files
    scn = Scene(filenames=glob('/Users/kenzatazi/Downloads/S3A_SL_1_RBT____20180822T000619_20180822T000919_20180822T015223_0179_035_016_3240_SVL_O_NR_003.SEN3/*'),
                reader='nc_slstr')

    # load datasets from input files

    def load_scene(scn):
        """ Loads the information from the netcdf files in the folder"""
        # scn.load(scn.available_dataset_names())
        scn.load(['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in',
                  'S9_in', 'bayes_an', 'bayes_in', 'cloud_an', 'longitude_an',
                  'latitude_an', 'solar_zenith_angle'])

    load_scene(scn)

    S1 = np.nan_to_num(scn['S1_an'].values)
    S2 = np.nan_to_num(scn['S2_an'].values)
    S3 = np.nan_to_num(scn['S3_an'].values)
    #S4= np.nan_to_num(scn['S4_an'].values)
    #S5= np.nan_to_num(scn['S5_an'].values)
    #S6= np.nan_to_num(scn['S6_an'].values)
    #S7= np.nan_to_num(np.array(scn['S7_in'][:-1]))
    #S8= np.nan_to_num(np.array(scn['S8_in'][:-1]))
    #S9= np.nan_to_num(np.array(scn['S9_in'][:-1]))

    bayes_mask = create_mask(scn, 'bayes_in')
    #emp_mask= create_mask(scn, 'cloud_an')

    # single channel images

    # channel_arrays=[S1, S2, S3, S4, S5, S6, S7, S8, S9]

    # for i in channel_arrays:
    #    plt.figure()
    #    plt.imshow(i, 'gray')

    # false color image

    fc = false_color_image(S3, S2, S1, plot=True)
    mask(bayes_mask, 'Baseyian mask', S1)
    #mask(emp_mask,'Empirical mask', S1)

    plt.show()
