# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:53:29 2018

@author: tomzh
"""
from Collocation2 import match_directory
import os
from SaveMatchedPixels import get_file_pairs, process_all, add_dist_col, add_time_col
from tqdm import tqdm
from FileDownloader import download_matches, Calipso_download


Home_directory = "/home/hep/trz15/Masters_Project"
NASA_FTP_directory = "48fcaa55-08b9-4eb8-b0e1-815769c43a2f"
calipso_directory = "/vols/lhcb/egede/cloud/Calipso/1km/2018/03/"
SLSTR_target_directory = "/vols/lhcb/egede/cloud/SLSTR/2018/03"
MatchesFilename = "Matches8.txt"
pkl_output_name = "Mar.pkl"
timewindow = 20

# Download Calipso file from NASA
Calipso_download(NASA_FTP_directory, calipso_directory)

# Find the SLSTR filenames which match the Calipso Filename
print("Beginning matching...")

os.chdir(Home_directory)
match_directory(calipso_directory, MatchesFilename, timewindow)

# Download the files found by match_directory
failed_downloads = download_matches(MatchesFilename, SLSTR_target_directory)

# Find matching pixels and store in pkl file
Cpaths, Spaths = get_file_pairs(
    calipso_directory, SLSTR_target_directory, MatchesFilename, failed_downloads)
df = process_all(Spaths, Cpaths, pkl_output_name)
df['Profile_Time'] += 725846390.0
df = add_dist_col(df)
df = add_time_col(df)
processed_pkl_name = pkl_output_name[:-4] + "P1.pkl"
df.to_pickle(processed_pkl_name)
