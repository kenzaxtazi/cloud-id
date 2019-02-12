
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import os

from Collocation import match_directory
from FileDownloader import NASA_download, download_matches
from SaveMatchedPixels import (add_dist_col, add_time_col, get_file_pairs,
                               process_all)

Home_directory = "/home/hep/trz15/Masters_Project"
NASA_FTP_directory = "528836af-a8ec-45eb-8391-8d24324fe1b6"
calipso_directory = ""
CATS_directory = "/vols/lhcb/egede/cloud/CATS/2017/08/"
SLSTR_target_directory = "/vols/lhcb/egede/cloud/SLSTR/2017/08"
MatchesFilename = "Matches12.txt"
pkl_output_name = "Aug17.pkl"
timewindow = 20
creds_path = '/home/hep/trz15/Masters_Project/credentials.txt'
processing_label = 'P3'

# Download Calipso file from NASA
NASA_download(NASA_FTP_directory, calipso_directory, CATS_directory)

# Find the SLSTR filenames which match the Calipso Filename
print("Beginning matching...")

os.chdir(Home_directory)
if calipso_directory != "":
    match_directory(calipso_directory, MatchesFilename, timewindow)
elif CATS_directory != "":
    match_directory(CATS_directory, MatchesFilename, timewindow)

print("File matching complete")

# Download the files found by match_directory
failed_downloads = download_matches(
    MatchesFilename, SLSTR_target_directory, creds_path)

# Find matching pixels and store in pkl file
Cpaths, Spaths = get_file_pairs(
    SLSTR_target_directory, MatchesFilename, failed_downloads, calipso_directory, CATS_directory)
df = process_all(Spaths, Cpaths, pkl_output_name)
df['Profile_Time'] += 725846390.0
df = add_dist_col(df)
df = add_time_col(df)
processed_pkl_name = pkl_output_name[:-4] + processing_label + ".pkl"
df.to_pickle(processed_pkl_name)

# Prepare a summary to print
with open(MatchesFilename, 'r') as file:
    data = file.readlines()

NumMatches = len(data)
NumPixels = len(df)
NumFailedDownloads = len(failed_downloads)

# Print summary
print()
print()
print('####################################################################')
print()
if calipso_directory != "":
    print("Processing of " + calipso_directory + " complete")
if CATS_directory != "":
    print("Processing of " + CATS_directory + " complete")

print("Number of Matches: " + str(NumMatches))
print("Number of Failed Downloads: " + str(NumFailedDownloads))
print("Number of Pixels Saved: " + str(NumPixels))
