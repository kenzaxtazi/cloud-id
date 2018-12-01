# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:53:29 2018

@author: tomzh
"""
from Collocation2 import match_directory, ESA_download
from ftplib import FTP
import os
from SaveMatchedPixels import get_file_pairs, process_all, add_dist_col, add_time_col
from tqdm import tqdm
from CEDAdownloadmatches import CEDA_download

Home_directory = "/home/hep/trz15/Masters_Project"
NASA_FTP_directory = "8aff26d6-6b5a-4544-ac03-bdddf25d7bbb"
calipso_directory = "/vols/lhcb/egede/cloud/Calipso/1km/2018/01/"
SLSTR_target_directory = "/vols/lhcb/egede/cloud/SLSTR/2018/01"
MatchesFilename = "Matches6.txt"
pkl_output_name = "Jan.pkl"
timewindow = 20

ftp = FTP('xfr140.larc.nasa.gov')
ftp.login('anonymous', 'trz15@imperial.ac.uk')
ftp.cwd(NASA_FTP_directory)
available_files = ftp.nlst()
files_to_download = [str(i) for i in available_files if str(i)[-1] == 'f']


os.chdir(calipso_directory)
for i in tqdm(files_to_download):
    ftp.retrbinary("RETR " + str(i), open(str(i), "wb").write)


# Find the SLSTR filenames which match the Calipso Filename
os.chdir(Home_directory)

match_directory(calipso_directory, MatchesFilename, timewindow)

# Download the files found by match_directory
with open(MatchesFilename, 'r') as file:
    data = file.readlines()
    
download_urls = [i.split(',')[2].strip() for i in data]
Sfilenames = [i.split(',')[1] for i in data]

# Attempt to download from CEDA's FTP server
failed_downloads = CEDA_download(MatchesFilename, SLSTR_target_directory)

downloaded_SLSTR_files = os.listdir(SLSTR_target_directory)

remaining_downloads = []

for i in range(len(Sfilenames)):
    if Sfilenames[i] + ".SEN3" not in downloaded_SLSTR_files:
        remaining_downloads.append(download_urls[i])

failed_downloads = ESA_download(remaining_downloads, SLSTR_target_directory)


if len(failed_downloads) > 0:
    DownloadedMatchesFilename = MatchesFilename[:-4] + 'b.txt'
    with open(DownloadedMatchesFilename, 'w') as file:
        for i in range(len(data)):
            if i not in failed_downloads:
                file.write(data[i])
    Cpaths, Spaths = get_file_pairs(calipso_directory, SLSTR_target_directory, DownloadedMatchesFilename)

else:
    Cpaths, Spaths = get_file_pairs(calipso_directory, SLSTR_target_directory, MatchesFilename)
df = process_all(Spaths, Cpaths, pkl_output_name)
df['Profile_Time'] += 725846390.0
df = add_dist_col(df)
df = add_time_col(df)
processed_pkl_name = pkl_output_name[:-4] + "P1.pkl"
df.to_pickle(processed_pkl_name)
