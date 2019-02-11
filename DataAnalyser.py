
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import DataLoader as DL
import DataPreparation as dp
import FileDownloader as FD
import Visualisation as Vis
from FFN import FFN


@pd.api.extensions.register_dataframe_accessor("da")
class DataAnalyser():
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def model_agreement(self, model, MaxDist=None, MaxTime=None, num_inputs=24):
        # Add useful columns to dataframe
        if MaxDist != None:
            self._obj = self._obj[self._obj['Distance'] < MaxDist]
        if MaxTime != None:
            self._obj = self._obj[abs(self._obj['TimeDiff']) < MaxTime]

        inputs = self._obj.dp.get_inputs(num_inputs)
        model = FFN(model)
        model.Load()

        output_labels = model.model.predict_label(inputs)
        output_con = model.model.predict(inputs)

        self._obj['Labels'] = pd.Series(
            output_labels[:, 0], index=self._obj.index)
        self._obj['Label_Confidence'] = pd.Series(
            output_con[:, 0], index=self._obj.index)

        self._obj = self._obj.dp.make_CTruth_col()

        self._obj['Agree'] = self._obj['CTruth'] != self._obj['Labels']
        return(self._obj)

    def get_bad_classifications(self):
        """Given a processed dataframe which has model predictions, produce dataframe with poorly classified pixels"""
        bad = self._obj[(self._obj['Agree'] == False) | (
            (self._obj['Label_Confidence'] < 0.7) & (self._obj['Label_Confidence'] > 0.3))]
        return(bad)

    def make_confidence_hist(self, model='Net1_FFN_v4', MaxDist=500, MaxTime=1200):
        """
        Makes a histogram of the model confidence for correctly and incorrectly classified pixels in a given directory or .pkl file.

        Parameters
        ----------
        model: str
            Name of a FFN model saved in the Models/ subdirectory
            Default is 'Net1_FFN_v4'

        MaxDist: int or float
            Maximum accepted distance between collocated pixels in dataframe to consider
            Default is 500

        MaxTime: int or float
            Maximum accepted time difference between collocated pixels in dataframe to consider
            Default is 1200
        """
        self._obj = self.model_agreement(model, MaxDist, MaxTime)

        wrong = self._obj[self._obj['Agree'] == False]

        bconfidence = wrong['Label_Confidence'].values
        tconfidence = self._obj['Label_Confidence'].values

        plt.hist(tconfidence, 250, label='Certainty of model for all predictions')
        plt.hist(bconfidence, 250,
                 label='Certainty of model for incorrect predictions')
        plt.legend()
        plt.title('Histogram of model prediction certainty for collocated data')
        plt.xlim([0, 1])
        plt.show()

    def plot_pixels(self, model='Net1_FFN_v4', MaxDist=500, MaxTime=1200):
        """
        Plots the correctly and incorrectly classified pixels in a given directory or .pkl file.

        Parameters
        ----------
        model: str
            Name of a FFN model saved in the Models/ subdirectory
            Default is 'Net1_FFN_v4'

        MaxDist: int or float
            Maximum accepted distance between collocated pixels in dataframe to consider
            Default is 500

        MaxTime: int or float
            Maximum accepted time difference between collocated pixels in dataframe to consider
            Default is 1200
        """
        self._obj = self.model_agreement(model, MaxDist, MaxTime)

        Vis.plot_poles(self._obj['latitude_an'].values,
                       self._obj['longitude_an'].values, self._obj['Agree'].values)

    def get_contextual_dataframe(self, contextlength=50, download_missing=False):
        """Given a dataframe of poorly classified pixels, produce dataframe with neighbouring S1 pixel values"""
        # List of all unique SLSTR files in the dataframe
        Sfiles = list(set(self._obj['Sfilename']))

        out = pd.DataFrame()

        if download_missing is True:
            ftp = FD.FTPlogin()

        for Sfile in tqdm(Sfiles):

            # Load the rows of the dataframe for a SLSTR file
            Sdf = self._obj[self._obj['Sfilename'] == Sfile]

            # Get the indices of the pixels
            Indices = Sdf[['RowIndex', 'ColIndex']].values

            # Get the path to the SLSTR file on the local machine
            Spath = DL.get_SLSTR_path(Sfile)

            # If the file is not on the local machine
            if os.path.exists(Spath) is False:

                if download_missing is True:
                    # Download the file
                    tqdm.write(Sfile + ' not found locally...')
                    tqdm.write('Downloading...')

                    Year = Sfile[16:20]
                    Month = Sfile[20:22]
                    Day = Sfile[22:24]

                    CEDApath = Year + '/' + Month + '/' + Day + '/' + Sfile + '.zip'

                    DestinationPath = '/vols/lhcb/egede/cloud/SLSTR/' + Year + '/' + Month + '/'

                    download_status = FD.FTPdownload(
                        ftp, CEDApath, DestinationPath)
                    if download_status == 1:
                        tqdm.write('Download failed, skipping...')
                        continue
                else:
                    tqdm.write(Sfile + ' not found locally...')
                    print('Skipping...')
                    continue

            coords = []

            for i in range(len(Indices)):
                x0, y0 = Indices[i]
                coords.append(dp.get_coords(x0, y0, contextlength, True))

            if len(coords) == 0:
                return(pd.DataFrame())

            scn = DL.scene_loader(Spath)
            scn.load(['S1_an'])
            S1 = np.array(scn['S1_an'].values)

            data = []

            for pixel in coords:
                pixel_data = []
                for arm in pixel:
                    xs = [i[0] for i in arm]
                    ys = [i[1] for i in arm]
                    arm_data = S1[xs, ys]
                    pixel_data.append(arm_data)
                data.append(pixel_data)

            SfileList = [Sfile] * len(data)
            Positions = list(Indices)

            newdf = pd.DataFrame(
                {'Sfilename': SfileList, 'Pos': Positions, 'Star_array': data})

            out = out.append(newdf, ignore_index=True, sort=True)

        return(out)
