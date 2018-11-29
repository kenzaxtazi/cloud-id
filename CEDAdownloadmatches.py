# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:44:37 2018

@author: tomzh
"""

import DataLoader as DL
import os
from tqdm import tqdm
import zipfile

def CEDA_download(MatchesFilename='Matches.txt', SLSTR_target_directory="/vols/lhcb/egede/cloud/SLSTR/2018/04"):

    with open('Matches.txt', 'r') as file:
        data = file.readlines()
        
    Sfiles = [i.split(',')[1] for i in data]
    Sfiles = list(set(Sfiles))
    
    ftp = DL.FTPlogin()
        
    startdir = os.getcwd()
    os.chdir(SLSTR_target_directory)
    
    q = os.listdir()
    
    Sfiles1 = []
    
    for i in range(len(Sfiles)):
        if Sfiles[i] + ".SEN3" not in q:
            Sfiles1.append(Sfiles[i])
            
    Sfiles2 = []
    
    for i in tqdm(range(len(Sfiles1))):
        path = Sfiles1[i]
        path = path[16:20] + '/' + path[22:24] + '/' + path[:] + '.zip'
        Sfiles2.append(path)
            
    for i in tqdm(range(len(Sfiles2))):
        targetfile = Sfiles2[i]
        downloadedfile = str(Sfiles1[i] + ".zip")
        tqdm.write('Downloading' + str(targetfile))
        try:
            ftp.retrbinary("RETR " + targetfile, open(downloadedfile, "wb").write)
            z = zipfile.ZipFile(downloadedfile)
            z.extractall()
            os.remove(downloadedfile)
        except:
            tqdm.write('Error downloading' + file)
            try:
                os.remove(downloadedfile)
            except FileNotFoundError:
                pass
            
    os.chdir(startdir)
    
if __name__ == "__main__":
    CEDA_download()