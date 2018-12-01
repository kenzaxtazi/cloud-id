# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:44:37 2018

@author: tomzh
"""

import DataLoader as DL
import os
from tqdm import tqdm
import zipfile


def CEDA_download(MatchesFilename='Matches4.txt', SLSTR_target_directory="/vols/lhcb/egede/cloud/SLSTR/2018/08"):
    """Function to download SLSTR files in a Matches.txt from CEDA's FTP server""" 
    failed_downloads = []

    with open(MatchesFilename, 'r') as file:
        data = file.readlines()

    # Get list of unique SLSTR files from Matches file
    Sfiles = [i.split(',')[1] for i in data]
    Sfiles = list(set(Sfiles))

    ftp = DL.FTPlogin()

    startdir = os.getcwd()
    os.chdir(SLSTR_target_directory)

    # List of files which are already downloaded
    q = os.listdir()

    # List of files which have not been downloaded yet
    Sfiles1 = []

    for i in range(len(Sfiles)):
        if Sfiles[i] + ".SEN3" not in q:
            Sfiles1.append(Sfiles[i])

    # List of file paths on CEDA which need to be downloaded
    Sfiles2 = []

    for i in tqdm(range(len(Sfiles1))):
        path = Sfiles1[i]
        path = path[16:20] + '/' + path[20:22] + '/' +  path[22:24] + '/' + path[:] + '.zip'
        Sfiles2.append(path)

    # Complete the downloads
    for i in tqdm(range(len(Sfiles2))):
        targetfile = Sfiles2[i] # File on CEDA to download
        downloadedfile = str(Sfiles1[i] + ".zip") # Name of file when downloaded
        tqdm.write('Downloading ' + str(targetfile))
        try:
            ftp.retrbinary("RETR " + targetfile,
                           open(downloadedfile, "wb").write)
            z = zipfile.ZipFile(downloadedfile)
            z.extractall()
            os.remove(downloadedfile)
        except:
            tqdm.write('Error downloading ' + str(targetfile))
            failed_downloads.append(Sfiles1[i])
            try:
                os.remove(downloadedfile)
            except FileNotFoundError:
                pass

    os.chdir(startdir)
    return(failed_downloads)

if __name__ == "__main__":
    CEDA_download()
    