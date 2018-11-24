# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:47:29 2018

@author: tomzh
"""

import os
from Collocation2 import ESA_download

with open('Matches.txt', 'r') as file:
    data = file.readlines()
    
q = os.listdir("/vols/lhcb/egede/cloud/SLSTR/2018/04")

downloads = [i.split(',')[2].strip() for i in data]
Sfiles = [i.split(',')[1] for i in data]

downloads1 = []

for i in range(len(Sfiles)):
    if Sfiles[i] + ".SEN3" not in q:
        downloads1.append(downloads[i])

ESA_download(downloads1, "/vols/lhcb/egede/cloud/SLSTR/2018/04")