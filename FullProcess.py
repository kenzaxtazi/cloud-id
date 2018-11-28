# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:53:29 2018

@author: tomzh
"""
from Collocation2 import match_directory, ESA_download
from ftplib import FTP
import os
from SaveMatchedPixels import get_file_pairs, process_all, add_dist_col, add_time_col



NASA_FTP_directory = ""
calipso_directory = ""
SLSTR_target_directory = ""
MatchesFilename = ""
pkl_output_name = ""
timewindow = 10

ftp = FTP('xfr140.larc.nasa.gov')
ftp.login('anonymous', 'trz15@imperial.ac.uk')
available_files = ftp.nlst()
files_to_download = [str(i) for i in available_files if str(i)[-1] == 'f']


os.chdir(calipso_directory)
for i in files_to_download:
    ftp.retrbinary("RETR " + str(i), open(str(i), "wb").write)


# Find the SLSTR filenames which match the Calipso Filename
match_directory(calipso_directory, MatchesFilename, timewindow)

# Download the files found by match_directory
with open(MatchesFilename, 'r') as file:
    data = file.readlines()
    
downloaded_SLSTR_files = os.listdir(SLSTR_target_directory)

download_urls = [i.split(',')[2].strip() for i in data]
Sfilenames = [i.split(',')[1] for i in data]

remaining_downloads = []

for i in range(len(Sfilenames)):
    if Sfilenames[i] + ".SEN3" not in downloaded_SLSTR_files:
        remaining_downloads.append(download_urls[i])

ESA_download(remaining_downloads, "SLSTR_target_directory")

Cpaths, Spaths = get_file_pairs(calipso_directory, SLSTR_target_directory, MatchesFilename)
df = process_all(Spaths, Cpaths, pkl_output_name)
df['Profile_Time'] += 725846390.0
df = add_dist_col(df)
df = add_time_col(df)
processed_pkl_name = pkl_output_name + "P1"
df.to_pickle(processed_pkl_name)
