# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:44:37 2018

@author: tomzh
"""

import DataLoader as DL
import os
from ftplib import FTP
from tqdm import tqdm
import zipfile

with open('Matches.txt', 'r') as file:
    data = file.readlines()
    
Sfiles = [i.split(',')[1] for i in data]


ftp = DL.FTPlogin()
    
startdir = os.getcwd()
os.chdir("/vols/lhcb/egede/cloud/SLSTR/2018/04")

q = os.listdir()

Sfiles1 = []

for i in range(len(Sfiles)):
    if Sfiles[i] + ".SEN3" not in q:
        path = Sfiles[i]
        path = path[16:20] + '/' + path[20:22] + '/' + path[22:24] + '/' + path[:] + ".zip"
        Sfiles1.append(path)
        
for i in tqdm(range(len(Sfiles1))):
    targetfile = Sfiles1[i]
    downloadedfile = Sfiles[i] + ".zip"
    tqdm.write('Downloading' + file)
    try:
        ftp.retrbinary("RETR " + targetfile, open(downloadedfile, "wb").write)
        z = zipfile.ZipFile(downloadedfile)
        z.extractall()
        os.remove(downloadedfile)
    except:
        print('Error downloading' + file)
        
os.chdir(startdir)