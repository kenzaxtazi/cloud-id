##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

import DataLoader as DL
import DataPreparation as dp
import ModelEvaluation as me
import Visualisation as Vis
from FFN import FFN
from CNN import CNN
from SuperModel import SuperModel

matplotlib.rcParams.update({'errorbar.capsize': 0.15})


@pd.api.extensions.register_dataframe_accessor("da")
class DataAnalyser():
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.model = None
        self.shuffled_channel = None

    def _model_applied(self):
        """Raise error if Agree column is not in dataframe"""
        if 'Agree' not in self._obj.columns:
            raise AttributeError(
                'No model has been applied to this dataframe.'
                ' See df.da.model_agreement')

    def model_agreement(self, model, modeltype='FFN', verbose=False, MaxDist=None, MaxTime=None):
        """
        Apply a model to the dataframe and add model output to rows

        Adds the direct output of the model into the 'Labels' and
        'Label_Confidence' columns, in addition the 'Agree' column shows
        whether the model result agrees with the Calipso truth.

        Parameters
        ----------
        model: str
            Name of model to use. If using a model on disk, it should be saved in the Models folder.

        Returns
        ----------
        None
        """

        if MaxDist is not None:
            self._obj = self._obj[self._obj['Distance'] < MaxDist]
        if MaxTime is not None:
            self._obj = self._obj[abs(self._obj['TimeDiff']) < MaxTime]

        if isinstance(model, str):
            self.model = model

            if modeltype == 'FFN':
                model = FFN(model)
            if modeltype == 'CNN':
                model = CNN(model)
            if modeltype == 'SuperModel':
                model = SuperModel(model)

            model.Load(verbose=verbose)

        elif isinstance(model, FFN):
            pass
        elif isinstance(model, CNN):
            pass
        elif isinstance(model, SuperModel):
            pass

        if modeltype == 'FFN':
            num_inputs = model.para_num
            inputs = self._obj.dp.get_ffn_inputs(num_inputs)
            output_labels = model.model.predict_label(inputs)
            output_con = model.model.predict(inputs)

        if modeltype == 'CNN':
            inputs = self._obj.dp.get_cnn_inputs()
            output_labels = model.model.predict_label(inputs)
            output_con = model.model.predict(inputs)

        if modeltype == 'SuperModel':
            num_inputs = model.para_num
            ffninputs = self._obj.dp.get_ffn_inputs(num_inputs)
            predictions1 = model.FFN.Predict(ffninputs)[:, 0]
            labels1 = model.FFN.model.predict_label(ffninputs)[:, 0]

            # boolean mask of bad predictions
            bad = abs(predictions1 - 0.5) < 0.25
            goodindices = np.where(bad == False)[0]
            badindices = np.where(bad == True)[0]
            cnninputs = self._obj[badindices].dp.get_cnn_inputs()
            cnninputs = dp.star_padding(cnninputs)

            # Feeding all of the inputs at once can cause a memory error
            # Instead split into chunks of 10,000
            chunkedcnninputs = [cnninputs[i: i + 10000]
                                for i in range(0, len(cnninputs), 10000)]

            predictions2 = []
            labels2 = []

            for i in range(len(chunkedcnninputs)):
                predictions2.extend(model.CNN.model.predict(
                    chunkedcnninputs[i])[:, 0])
                labels2.extend(model.CNN.model.predict_label(
                    chunkedcnninputs[i])[:, 0])

            finallabels = np.zeros(len(self._obj))
            finallabels[goodindices] = labels1[goodindices]
            finallabels[badindices] = labels2

            finalpredictions = np.zeros(len(self._obj))
            finalpredictions[goodindices] = predictions1[goodindices]
            finalpredictions[badindices] = predictions2

            output_labels = finallabels
            output_con = finalpredictions

        self._obj['Labels'] = pd.Series(output_labels[:, 0], index=self._obj.index)
        self._obj['Label_Confidence'] = pd.Series(output_con[:, 0], index=self._obj.index)

        self._obj = self._obj.dp.make_CTruth_col()

        self._obj['Agree'] = self._obj['CTruth'] != self._obj['Labels']

    def shuffled_model_agreement(self, model, channel_name, verbose=False, MaxDist=None, MaxTime=None):
        """
        Apply a model to the dataframe and add model output to rows

        Adds the direct output of the model into the 'Labels' and
        'Label_Confidence' columns, in addition the 'Agree' column shows
        whether the model result agrees with the Calipso truth.

        Parameters
        ----------
        model: str
            Name of model to use. If using a model on disk, it should be saved in the Models folder.

        Returns
        ----------
        None
        """

        self.shuffled_channel = channel_name

        if MaxDist is not None:
            self._obj = self._obj[self._obj['Distance'] < MaxDist]
        if MaxTime is not None:
            self._obj = self._obj[abs(self._obj['TimeDiff']) < MaxTime]

        if isinstance(model, str):
            self.model = model

            model = FFN(model)
            model.Load(verbose=verbose)

        elif isinstance(model, FFN):
            pass

        channel_indices = {
            'S1_an': 0,
            'S2_an': 1,
            'S3_an': 2,
            'S4_an': 3,
            'S5_an': 4,
            'S6_an': 5,
            'S7_in': 6,
            'S8_in': 7,
            'S9_in': 8,
            'satellite_zenith_angle': 9,
            'solar_zenith_angle': 10,
            'latitude_an': 11,
            'longitude_an': 12}

        num_inputs = model.para_num
        inputs = self._obj.dp.get_ffn_inputs(num_inputs)
        shuffled_inputs = np.column_stack((inputs[:, :channel_indices[channel_name]],
                                           np.random.permutation(
                                               inputs[:, channel_indices[channel_name]]),
                                           inputs[:, channel_indices[channel_name] + 1:]))
        output_labels = model.model.predict_label(inputs)
        output_con = model.model.predict(inputs)
        shuffled_output_con = model.model.predict(shuffled_inputs)

        self._obj['Labels'] = pd.Series(
            output_labels[:, 0], index=self._obj.index)
        self._obj['Label_Confidence'] = pd.Series(
            output_con[:, 0], index=self._obj.index)
        self._obj['Shuffled_Confidence'] = pd.Series(
            shuffled_output_con[:, 0], index=self._obj.index)

        self._obj = self._obj.dp.make_CTruth_col()

        self._obj['Agree'] = self._obj['CTruth'] != self._obj['Labels']

    def get_bad_classifications(self):
        """
        Given a dataframe with model predictions, return poor results.

        Returns
        ----------
        bad: pandas DataFrame
            Dataframe with rows where either the model disagrees with Calipso
            or model confidence is low.
        """
        self._model_applied()

        bad = self._obj[(self._obj['Agree'] is False) | (
            (self._obj['Label_Confidence'] < 0.7) & (self._obj['Label_Confidence'] > 0.3))]
        return(bad)

    def make_confidence_hist(self):
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
        self._model_applied()

        # Is false causes key error
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

    def plot_pixels(self, datacol='Agree'):
        """
        Plots the value of data for each pixel on map.

        Parameters
        ----------
        datacol: str
            Name of dataframe column to use for colouring pixels on map

        Returns
        ----------
        None
        """
        if datacol in ['Agree', 'CTruth', 'Labels', 'Label_Confidence']:
            self._model_applied()

        Vis.plot_poles(self._obj['latitude_an'].values,
                       self._obj['longitude_an'].values, self._obj[datacol].values)

    def plot_poles_gridded(self, datacol='Agree'):
        if datacol in ['Agree', 'CTruth', 'Labels', 'Label_Confidence']:
            self._model_applied()

        lat = self._obj['latitude_an'].values
        lon = self._obj['longitude_an'].values
        data = self._obj['Agree'].values

        lat = np.round_(lat, 1)
        lon = np.round_(lon, 1)

        pos = list(zip(lat, lon, data))

        upos = list(set(pos))
        upos.sort()

        cnt = Counter(pos)

        Tpos = list(zip(lat, lon, [True] * len(lat)))
        Fpos = list(zip(lat, lon, [False] * len(lat)))

        Tpos.sort()
        Fpos.sort()

        NTrues = []
        NFalses = []

        for i in Tpos:
            NTrues.append(cnt[i])

        for i in Fpos:
            NFalses.append(cnt[i])

        Means = []

        for i in range(len(NTrues)):
            Means.append(NTrues[i] / (NTrues[i] + NFalses[i]))

        ulat = [i[0] for i in upos]
        ulon = [i[1] for i in upos]

        Vis.plot_poles(ulat, ulon, Means, 1.5)

    def get_contextual_dataframe(self, pklname, contextlength=50, square=False):
        """Given a dataframe of poorly classified pixels, produce dataframe with neighbouring S1 pixel values"""
        # List of all unique SLSTR files in the dataframe
        Sfiles = list(set(self._obj['Sfilename']))

        out = pd.DataFrame()

        for i, Sfile in enumerate(tqdm(Sfiles)):

            # Load the rows of the dataframe for a SLSTR file
            Sdf = self._obj[self._obj['Sfilename'] == Sfile]

            # Get the indices of the pixels
            Indices = Sdf[['RowIndex', 'ColIndex']].values

            # Get the path to the SLSTR file on the local machine
            Spath = DL.get_SLSTR_path(Sfile)

            # If the file is not on the local machine
            if os.path.exists(Spath) is False:
                tqdm.write(Sfile + ' not found locally...')
                tqdm.write('Skipping...')
                continue

            if square is False:
                coords = []

                for j in range(len(Indices)):
                    x0, y0 = Indices[j]
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
                        xs = [j[0] for j in arm]
                        ys = [j[1] for j in arm]
                        arm_data = S1[xs, ys]
                        pixel_data.append(arm_data)
                    data.append(pixel_data)

                SfileList = [Sfile] * len(data)
                Positions = list(Indices)

                newdf = pd.DataFrame(
                    {'Sfilename': SfileList, 'Pos': Positions, 'Star_array': data})

                out = out.append(newdf, ignore_index=True, sort=True)

                if i % 25 == 0 or i == len(Sfiles) - 1:
                    if i == 0:
                        out.to_pickle(pklname)
                    else:
                        temp = pd.read_pickle(pklname)
                        temp = temp.append(out)
                        temp.to_pickle(pklname)
                        out = pd.DataFrame()

            else:
                scn = DL.scene_loader(Spath)
                scn.load(['S1_an'])
                S1 = np.zeros((2410, 3010))
                try:
                    S1[5:2405, 5:3005] = np.array(scn['S1_an'].values)
                except ValueError:
                    tqdm.write('Skipping improperly shaped array')
                    tqdm.write(Spath)
                    continue

                data = []

                for j in range(len(Indices)):
                    y0, x0 = Indices[j]
                    y0 += 5
                    x0 += 5
                    data.append(S1[y0 - 5:y0 + 6, x0 - 5:x0 + 6])

                SfileList = [Sfile] * len(data)
                Positions = list(Indices)

                newdf = pd.DataFrame(
                    {'Sfilename': SfileList, 'Pos': Positions, 'Star_array': data})

                out = out.append(newdf, ignore_index=True, sort=True)

                if i % 25 == 0 or i == len(Sfiles) - 1:
                    if i == 0:
                        out.to_pickle(pklname)
                    else:
                        temp = pd.read_pickle(pklname)
                        temp = temp.append(out)
                        temp.to_pickle(pklname)
                        out = pd.DataFrame()

    def AUC_timediff(self, seed=2553149187, validation_frac=0.15):
        """
        Produces a histogram of accuraccy as a function of the time difference between
        the data take by SLSTR and CALIOP instruments

        Parameters
        -----------
        model: model object

        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        None
        """
        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        time_slices = np.linspace(0, 1401, 15)
        aucs = []
        N = []

        for t in time_slices:

            sliced_df = valdf[valdf['TimeDiff'].between(t, t + 100)]

            if len(sliced_df) > 0:
                auc = metrics.roc_auc_score((sliced_df['CTruth'].values).astype('int'),
                                            sliced_df['Label_Confidence'].values)
                aucs.append(auc)
                N.append(len(sliced_df))
            else:
                aucs.append(0)
                N.append(0)

        plt.figure('AUC vs time difference')
        plt.title('AUC as a function of time difference')
        plt.xlabel('Absolute time difference (s)')
        plt.ylabel('AUC')
        plt.bar(time_slices, aucs, width=100, align='edge',
                color='lightcyan', edgecolor='lightseagreen', yerr=(np.array(aucs) / np.array(N))**(0.5))
        plt.show()

    def accuracy_timediff_for_broken_cloud(self, seed=2553149187, validation_frac=0.15):
        """
        Produces a histogram of accuraccy as a function of the time difference between
        the data take by SLSTR and CALIOP instruments

        Parameters
        -----------
        model: model object

        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        None
        """
        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]
        valdf['FCF_RightShift9'] = pd.Series(valdf['Feature_Classification_Flags'].values >> 9,
                                             index=valdf.index)

        # separate low broken cumulus
        cloudy_valdf = valdf[valdf['Feature_Classification_Flags'] & 7 == 2]
        brk_cml_valdf = cloudy_valdf[cloudy_valdf['FCF_RightShift9'] & 7 == 3]
        time_slices = np.linspace(0, 1401, 15)
        aucs = []
        N = []

        for t in time_slices:

            sliced_df = brk_cml_valdf[valdf['TimeDiff'].between(t, t + 100)]

            if len(sliced_df) > 0:
                auc = np.mean(sliced_df['Agree'])

                # metrics.roc_auc_score((sliced_df['CTruth'].values).astype('int'), (sliced_df['Label_Confidence'].values))
                aucs.append(auc)
                N.append(len(sliced_df))
            else:
                aucs.append(0)
                N.append(0)

        plt.figure('Accuracy vs time difference')
        plt.title('Accuracy as a function of time difference')
        plt.xlabel('Absolute time difference (s)')
        plt.ylabel('Accuracy')
        plt.bar(time_slices, aucs, width=100, align='edge',
                color='lightcyan', edgecolor='lightseagreen', yerr=(np.array(aucs) / np.array(N))**(0.5))
        plt.show()

    def AUC_sza(self, seed=2553149187, validation_frac=15):
        """
        Produces a histogram of accuraccy as a function of solar zenith angle

        Parameters
        -----------
        model: model object

        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        Matplotlib histogram
        """

        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        angle_slices = np.linspace(3, 55, 18)
        aucs = []
        N = []

        for a in angle_slices:

            sliced_df = valdf[valdf['satellite_zenith_angle'].between(
                a, a + 3)]

            if len(sliced_df) > 0:
                auc = metrics.roc_auc_score((sliced_df['CTruth'].values).astype('int'),
                                            (sliced_df['Label_Confidence'].values))
                aucs.append(auc)
                N.append(len(sliced_df))
            else:
                aucs.append(0)
                N.append(0)

        plt.figure('AUC vs satellite zenith angle')
        plt.title('AUC as a function of satellite zenith angle')
        plt.xlabel('Satellite zenith angle (deg)')
        plt.ylabel('AUC')
        plt.bar(angle_slices, aucs, width=3, align='edge', color='lavenderblush',
                edgecolor='thistle', ecolor='purple', yerr=(np.array(aucs) / np.array(N))**(0.5))
        plt.show()

    def AUC_ctype(self, seed=2553149187, validation_frac=0.15):
        """
        Produces a histogram of accuracy as a function of cloud type

        Parameters
        -----------
        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        Matplotlib histogram
        """
        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        bitmeanings = {
            'low overcast, transparent': 0,
            'low overcast, opaque': 1,
            'transition stratocumulus': 2,
            'low, broken cumulus': 3,
            'altocumulus (transparent)': 4,
            'altostratus (opaque)': 5,
            'cirrus (transparent)': 6,
            'deep convective (opaque)': 7}

        model_accuracies = []
        bayes_accuracies = []
        empir_accuracies = []

        N = []

        # Seperate clear flags
        cleardf = valdf[valdf['Feature_Classification_Flags'] & 7 != 2]
        n = len(cleardf)
        N.append(n)

        # Model accuracy
        model_accuracy = np.mean(cleardf['Agree'])
        model_accuracies.append(model_accuracy)

        # Bayesian mask accuracy
        bayes_labels = cleardf['bayes_in']
        bayes_labels[bayes_labels > 1] = 1
        bayes_accuracy = float(
            len(bayes_labels[bayes_labels == cleardf['CTruth']])) / float(n)
        bayes_accuracies.append(bayes_accuracy)

        # Empirical mask accuracy
        empir_labels = cleardf['cloud_an']
        empir_labels[empir_labels > 1] = 1
        empir_accuracy = float(
            len(empir_labels[empir_labels == cleardf['CTruth']])) / float(n)
        empir_accuracies.append(empir_accuracy)

        # Seperate cloudy flags
        cloudydf = valdf[valdf['Feature_Classification_Flags'] & 7 == 2]

        for cloud in bitmeanings:

            cloud_df = cloudydf[cloudydf['FCF_RightShift9']
                                & 7 == bitmeanings[cloud]]

            # Model accuracy
            n = len(cloud_df)
            model_accuracy = np.mean(cloud_df['Agree'])
            # print(str(surface) + ': ' + str(accuracy))

            # Bayesian mask accuracy
            bayes_labels = cloud_df['bayes_in']
            bayes_labels[bayes_labels > 1] = 1
            bayes_accuracy = float(
                len(bayes_labels[bayes_labels == cloud_df['CTruth']])) / float(n)

            # Empirical mask accuracy
            empir_labels = cloud_df['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_accuracy = float(
                len(empir_labels[empir_labels == cloud_df['CTruth']])) / float(n)

            model_accuracies.append(model_accuracy)
            bayes_accuracies.append(bayes_accuracy)
            empir_accuracies.append(empir_accuracy)
            N.append(n)

        names = ['Clear', 'Low overcast, transparent', 'Low overcast, opaque',
                 'Transition stratocumulus', 'Low, broken cumulus',
                 'Altocumulus (transparent)', 'Altostratus (opaque)',
                 'Cirrus (transparent)', 'Deep convective (opaque)']

        t = np.arange(len(names))

        plt.figure('Accuracy vs cloud type')
        plt.title('Accuracy as a function of cloud type')
        plt.ylabel('Accuracy')
        bars = plt.bar(t, model_accuracies, width=0.5, align='center', color='honeydew',
                       edgecolor='palegreen', yerr=(np.array(model_accuracies) / np.array(N))**(0.5),
                       tick_label=names, ecolor='g', capsize=3, zorder=1)
        circles = plt.scatter(t, bayes_accuracies, marker='o', zorder=2)
        stars = plt.scatter(t, empir_accuracies, marker='*', zorder=3)
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.xticks(rotation=90)
        plt.legend([bars, circles, stars], ['Model accuracy',
                                            'Bayesian mask accuracy',
                                            'Empirical mask accuracy'])
        plt.show()

    def AUC_stype(self, seed=2553149187, validation_frac=0.15):
        """
        Produces a histogram of accuracy as a function of surface type.

        Parameters
        -----------
        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        Matplotlib histogram.

        """

        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        bitmeanings = {
            'Coastline': 1,
            'Ocean': 2,
            'Tidal': 4,
            'Dry land': 24,
            'Inland water': 16,
            'Cosmetic': 256,
            'Duplicate': 512,
            'Day': 1024,
            'Twilight': 2048,
            'Snow': 8192}

        model_accuracies = []
        bayes_accuracies = []
        empir_accuracies = []
        aucs = []
        N = []

        for surface in bitmeanings:

            if surface != 'dry_land':
                surfdf = valdf[valdf['confidence_an']
                               & bitmeanings[surface] == bitmeanings[surface]]
            else:
                surfdf = valdf[valdf['confidence_an']
                               & bitmeanings[surface] == 8]

            # Model accuracy
            n = len(surfdf)
            model_accuracy = np.mean(surfdf['Agree'])
            auc = metrics.roc_auc_score((surfdf['CTruth'].values).astype('int'),
                                        (surfdf['Label_Confidence'].values))
            aucs.append(auc)

            # print(str(surface) + ': ' + str(accuracy))

            # Bayesian mask accuracy
            bayes_labels = surfdf['bayes_in']
            bayes_labels[bayes_labels > 1] = 1
            bayes_accuracy = float(
                len(bayes_labels[bayes_labels == surfdf['CTruth']])) / float(n)

            # Empirical mask accuracy
            empir_labels = surfdf['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_accuracy = float(
                len(empir_labels[empir_labels == surfdf['CTruth']])) / float(n)

            model_accuracies.append(model_accuracy)
            bayes_accuracies.append(bayes_accuracy)
            empir_accuracies.append(empir_accuracy)
            N.append(n)

        names = ['Coastline', 'Ocean', 'Tidal', 'Land', 'Inland water',
                 'Cosmetic', 'Duplicate', 'Day', 'Twilight', 'Snow']

        print(aucs)

        t = np.arange(len(names))

        plt.figure('Accuracy vs surface type')
        plt.title('Accuracy as a function of surface type')
        plt.ylabel('Accuracy')
        bars = plt.bar(t, model_accuracies, width=0.5, align='center', color='honeydew',
                       edgecolor='palegreen', yerr=(np.array(model_accuracies) / np.array(N))**(0.5),
                       tick_label=names, ecolor='g', capsize=3, zorder=1)
        circles = plt.scatter(t, bayes_accuracies, marker='o', zorder=2)
        stars = plt.scatter(t, empir_accuracies, marker='*', zorder=3)
        plt.yticks([0.50, 0.55, 0.60, 0.65, 0.70,
                    0.75, 0.80, 0.85, 0.90, 0.95])
        plt.xticks(rotation=45)
        plt.legend([bars, circles, stars], ['Model accuracy',
                                            'Bayesian mask accuracy',
                                            'Empirical mask accuracy'])
        plt.show()

    def confidence_ctype(self, seed=2553149187, validation_frac=0.15):
        """
        Produces two histograms of confidence as a function of cloud type for
        data points classified as cloudy and clear and one stacked bar chart
        with data points numbers.

        Parameters
        -----------
        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        Matplotlib histograms
        """
        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        clear_valdf = valdf[valdf['Labels'] == 1]
        cloudy_valdf = valdf[valdf['Labels'] == 0]

        bitmeanings = {
            'Low overcast, transparent': 0,
            'Low overcast, opaque': 1,
            'Transition stratocumulus': 2,
            'Low, broken cumulus': 3,
            'Altocumulus (transparent)': 4,
            'Altostratus (opaque)': 5,
            'Cirrus (transparent)': 6,
            'Deep convective (opaque)': 7}

        clear_probabilities = []
        cloudy_probabilities = []
        Ncloudy = []
        Nclear = []

        # seperate clear flags
        clear_cleardf = clear_valdf[clear_valdf['Feature_Classification_Flags'] & 7 != 2]
        clear_cloudydf = cloudy_valdf[cloudy_valdf['Feature_Classification_Flags'] & 7 != 2]
        clear_probabilities.append(
            np.mean(clear_cleardf['Label_Confidence'].values))
        cloudy_probabilities.append(
            np.mean(clear_cloudydf['Label_Confidence'].values))
        Ncloudy.append(len(clear_cloudydf))
        Nclear.append(len(clear_cleardf))

        # seperate cloudy flags
        cloudy_cleardf = clear_valdf[clear_valdf['Feature_Classification_Flags'] & 7 == 2]
        cloudy_cloudydf = cloudy_valdf[cloudy_valdf['Feature_Classification_Flags'] & 7 == 2]

        # new column with shifted feature classifcation flags to get cloud subtypes
        cloudy_cleardf['FCF_RightShift9'] = pd.Series(
            cloudy_cleardf['Feature_Classification_Flags'].values >> 9, index=cloudy_cleardf.index)
        cloudy_cloudydf['FCF_RightShift9'] = pd.Series(
            cloudy_cloudydf['Feature_Classification_Flags'].values >> 9, index=cloudy_cloudydf.index)

        for surface in bitmeanings:
            cleardf = cloudy_cleardf[cloudy_cleardf['FCF_RightShift9']
                                     & 7 == bitmeanings[surface]]
            cloudydf = cloudy_cloudydf[cloudy_cloudydf['FCF_RightShift9']
                                       & 7 == bitmeanings[surface]]

            clear_probabilities.append(
                np.mean(cleardf['Label_Confidence'].values))
            cloudy_probabilities.append(
                np.mean(cloudydf['Label_Confidence'].values))
            Ncloudy.append(len(cloudydf))
            Nclear.append(len(cleardf))

        names = ['Clear', 'Low overcast, Transparent', 'Low overcast, opaque', 'Transition stratocumulus', 'Low, broken cumulus',
                 'Altocumulus (transparent)', 'Altostratus (opaque)', 'Cirrus (transparent)', 'Deep convective (opaque)']

        t = np.arange(len(names))

        plt.figure('Average cloudy confidence vs cloud type')
        plt.title(
            'Average confidence as a function of cloud type for pixels classified as cloud')
        plt.ylabel('Average probability')
        plt.bar(t, cloudy_probabilities, width=0.5, align='center', color='lavender',
                edgecolor='plum', yerr=(np.array(cloudy_probabilities) / np.array(Ncloudy))**(0.5),
                tick_label=names, ecolor='purple', capsize=3)
        plt.xticks(rotation=90)

        plt.figure('Average clear confidence vs cloud type')
        plt.title(
            'Average confidence as a function of cloud type for pixels classified as clear')
        plt.ylabel('Average probability')
        plt.bar(t, clear_probabilities, width=0.5, align='center', color='lavender',
                edgecolor='plum', yerr=(np.array(clear_probabilities) / np.array(Nclear))**(0.5),
                tick_label=names, ecolor='purple', capsize=3)
        plt.xticks(rotation=90)

        plt.figure('Classification numbers vs cloud type')
        plt.title('Classification numbers as a function of cloud type')
        plt.ylabel('Number of data points')
        bars1 = plt.bar(t, Ncloudy, width=0.5, align='center', color='papayawhip',
                        edgecolor='bisque', tick_label=names, ecolor='orange')
        bars2 = plt.bar(t, Nclear, width=0.5, align='center', color='lightcyan',
                        edgecolor='lightskyblue', bottom=Ncloudy, tick_label=names, ecolor='skyblue')
        plt.xticks(rotation=90)
        plt.legend([bars1, bars2], [
                   'Predicted as cloudy', 'Predicted as clear'])
        plt.show()

    def confidence_stype(self, seed=2553149187, validation_frac=0.15):
        """
        Produces two histograms of confidence as a function of surface type for
        data points classified as cloudy and clear and one stacked bar chart
        with data points numbers.

        Parameters
        -----------
        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        Matplotlib histograms
        """

        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        clear_valdf = valdf[valdf['Labels'] == 1]
        cloudy_valdf = valdf[valdf['Labels'] == 0]

        bitmeanings = {
            'Coastline': 1,
            'Ocean': 2,
            'Tidal': 4,
            'Dry land': 24,
            'Inland water': 16,
            'Cosmetic': 256,
            'Duplicate': 512,
            'Day': 1024,
            'Twilight': 2048,
            'Snow': 8192}

        clear_probabilities = []
        cloudy_probabilities = []
        Ncloudy = []
        Nclear = []

        for surface in bitmeanings:

            if surface != 'Dry land':
                cleardf = clear_valdf[clear_valdf['confidence_an']
                                      & bitmeanings[surface] == bitmeanings[surface]]
                cloudydf = cloudy_valdf[cloudy_valdf['confidence_an']
                                        & bitmeanings[surface] == bitmeanings[surface]]
            else:
                cleardf = clear_valdf[clear_valdf['confidence_an']
                                      & bitmeanings[surface] == 8]
                cloudydf = cloudy_valdf[cloudy_valdf['confidence_an']
                                        & bitmeanings[surface] == 8]

            clear_probabilities.append(
                np.mean(cleardf['Label_Confidence'].values))
            cloudy_probabilities.append(
                np.mean(cloudydf['Label_Confidence'].values))
            Ncloudy.append(len(cloudydf))
            Nclear.append(len(cleardf))

        names = ['Coastline', 'Ocean', 'Tidal', 'Land', 'Inland water',
                 'Cosmetic', 'Duplicate', 'Day', 'Twilight', 'Snow']

        t = np.arange(len(names))

        plt.figure('Average cloudy confidence vs surface type')
        plt.title(
            'Average confidence as a function of surface type for pixels classified as cloud')
        plt.ylabel('Average probability')
        plt.bar(t, cloudy_probabilities, width=0.5, align='center', color='lavender',
                edgecolor='plum', yerr=(np.array(cloudy_probabilities) / np.array(Ncloudy))**(0.5),
                tick_label=names, ecolor='purple', capsize=3)
        plt.xticks(rotation=90)

        plt.figure('Average clear confidence vs surface type')
        plt.title(
            'Average confidence as a function of surface type for pixels classified as clear')
        plt.ylabel('Average probability')
        plt.bar(t, clear_probabilities, width=0.5, align='center', color='lavender',
                edgecolor='plum', yerr=(np.array(clear_probabilities) / np.array(Nclear))**(0.5),
                tick_label=names, ecolor='purple', capsize=3)
        plt.xticks(rotation=90)

        plt.figure('Classification numbers vs surface type')
        plt.title('Classification numbers as a function of surface type')
        plt.ylabel('Number of data points')
        bars1 = plt.bar(t, Ncloudy, width=0.5, align='center', color='papayawhip',
                        edgecolor='bisque', tick_label=names, ecolor='orange')
        bars2 = plt.bar(t, Nclear, width=0.5, align='center', color='lightcyan',
                        edgecolor='lightskyblue', bottom=Ncloudy, tick_label=names, ecolor='skyblue')
        plt.xticks(rotation=90)
        plt.legend([bars1, bars2], [
                   'Predicted as cloudy', 'Predicted as clear'])
        plt.show()

    def confidence_atype(self, seed=2553149187, validation_frac=0.15):
        """
        Produces a histograms analysing clear types.

        - Two histograms of confidence as a function of surface cloud types for
        data points classified as cloudy and clear.
        - One stacked bar chart with data points numbers.

        Parameters
        -----------
        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        Matplotlib histograms
        """
        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        clear_valdf = valdf[valdf['Labels'] == 1]
        cloudy_valdf = valdf[valdf['Labels'] == 0]

        bitmeanings = {
            'Not determined': 0,
            'Clear marine': 1,
            'Dust': 2,
            'Polluted continental': 3,
            'Clean continental': 4,
            'Polluted dust': 5,
            'Smoke': 6,
            'Other': 7}

        clear_probabilities = []
        cloudy_probabilities = []
        Ncloudy = []
        Nclear = []

        # seperate clear air and cloudy flags
        clearair_cleardf = clear_valdf[clear_valdf['Feature_Classification_Flags'] & 7 == 1]
        clearair_cloudydf = cloudy_valdf[cloudy_valdf['Feature_Classification_Flags'] & 7 == 1]

        cloudy_cleardf = clear_valdf[clear_valdf['Feature_Classification_Flags'] & 7 == 2]
        cloudy_cloudydf = cloudy_valdf[cloudy_valdf['Feature_Classification_Flags'] & 7 == 2]

        clear_probabilities.extend([np.mean(cloudy_cleardf['Label_Confidence'].values),
                                    np.mean(clearair_cleardf['Label_Confidence'].values)])
        cloudy_probabilities.extend([np.mean(cloudy_cloudydf['Label_Confidence'].values),
                                     np.mean(clearair_cloudydf['Label_Confidence'].values)])

        Nclear.extend([len(cloudy_cleardf), len(clearair_cleardf)])
        Ncloudy.extend([len(cloudy_cloudydf), len(clearair_cloudydf)])

        # seperate aerosol flags
        aero_cleardf = clear_valdf[clear_valdf['Feature_Classification_Flags'] & 7 == 3]
        aero_cloudydf = cloudy_valdf[cloudy_valdf['Feature_Classification_Flags'] & 7 == 3]

        print(aero_cleardf)
        print(aero_cloudydf)

        # new column with shifted feature classifcation flags to get cloud subtypes
        aero_cleardf['FCF_RightShift9'] = pd.Series(
            aero_cleardf['Feature_Classification_Flags'].values >> 9, index=aero_cleardf.index)
        aero_cloudydf['FCF_RightShift9'] = pd.Series(
            aero_cloudydf['Feature_Classification_Flags'].values >> 9, index=aero_cloudydf.index)

        print(aero_cleardf['FCF_RightShift9'])
        print(aero_cloudydf['FCF_RightShift9'])

        for surface in bitmeanings:
            cleardf = aero_cleardf[aero_cleardf['FCF_RightShift9']
                                   & 7 == bitmeanings[surface]]
            cloudydf = aero_cloudydf[aero_cloudydf['FCF_RightShift9']
                                     & 7 == bitmeanings[surface]]

            clear_probabilities.append(
                np.mean(cleardf['Label_Confidence'].values))
            cloudy_probabilities.append(
                np.mean(cloudydf['Label_Confidence'].values))
            Ncloudy.append(len(cloudydf))
            Nclear.append(len(cleardf))

        names = ['Cloud', 'Clear', 'Not determined', 'Clear marine', 'Dust', 'Polluted continental',
                 'Clean continental', 'Polluted dust', 'Smoke', 'Other']

        print(clear_probabilities)
        print(cloudy_probabilities)

        t = np.arange(len(names))

        plt.figure('Average cloudy probability vs aerosol type')
        plt.title('Average cloudy probability as a function of aerosol type')
        plt.ylabel('Average probability')
        plt.bar(t, cloudy_probabilities, width=0.5, align='center', color='lavender',
                edgecolor='plum', yerr=(np.array(cloudy_probabilities) / np.array(Ncloudy))**(0.5),
                tick_label=names, ecolor='purple', capsize=3)
        plt.xticks(rotation=90)

        plt.figure('Average clear probability vs aerosol type')
        plt.title('Average clear probability as a function of aerosol type')
        plt.ylabel('Average probability')
        plt.bar(t, clear_probabilities, width=0.5, align='center', color='lavender',
                edgecolor='plum', yerr=(np.array(clear_probabilities) / np.array(Nclear))**(0.5),
                tick_label=names, ecolor='purple', capsize=3)
        plt.xticks(rotation=90)

        plt.figure('Classification numbers vs cloud type')
        plt.title('Classification numbers as a function of cloud type')
        plt.ylabel('Number of data points')
        bars1 = plt.bar(t, Ncloudy, width=0.5, align='center', color='papayawhip',
                        edgecolor='bisque', tick_label=names, ecolor='orange')
        bars2 = plt.bar(t, Nclear, width=0.5, align='center', color='lightcyan',
                        edgecolor='lightskyblue', bottom=Ncloudy, tick_label=names, ecolor='skyblue')
        plt.xticks(rotation=90)
        plt.legend([bars1, bars2], [
                   'Predicted as cloudy', 'Predicted as clear'])
        plt.show()

    def ROC_stype(self, seed=2553149187, validation_frac=0.15):
        """
        Produces ROCs of relevant SLSTR surface types.

        Parameters
        -----------
        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        Matplotlib plots
        """
        self._model_applied()

        bayesian_df = self._obj[['BayesProb', 'CTruth']].dropna()
        self._obj.drop(columns='BayesProb')

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        bitmeanings = {
            'Coastline': 1,
            'Ocean': 2,
            'Tidal': 4,
            'Dry land': 24,
            'Inland water': 16,
            'Cosmetic': 256,
            'Duplicate': 512,
            'Day': 1024,
            'Twilight': 2048,
            'Snow': 8192}

        for surface in bitmeanings:

            if surface != 'Dry Land':
                surfdf = valdf[valdf['confidence_an']
                               & bitmeanings[surface] == bitmeanings[surface]]
            else:
                surfdf = valdf[valdf['confidence_an']
                               & bitmeanings[surface] == 8]

            # Truth
            truth = surfdf['CTruth']
            truth_onehot = np.vstack((truth, ~truth)).T

            # Model
            model_confidence = surfdf['Label_Confidence']
            model_onehot = np.vstack(
                (model_confidence, 1 - model_confidence)).T

            # Bayesian mask
            bayes_labels = surfdf['bayes_in']
            bayes_labels[bayes_labels > 1] = 1
            bayes_onehot = np.vstack((bayes_labels, ~bayes_labels)).T

            # Bayesian prob and truth
            # bayes_p = surfdf['BayesProb']
            bayes_p_onehot = np.vstack((bayesian_df['BayesProb'], 1 - bayesian_df['BayesProb'])).T
            bayes_t_onehot = np.vstack((bayesian_df['CTruth'],  ~bayesian_df['CTruth'])).T

            # Empirical mask
            empir_labels = surfdf['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_onehot = np.vstack((empir_labels, ~empir_labels)).T

            #print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

            me.ROC(model_onehot, truth_onehot, bayes_mask=bayes_onehot,
                   emp_mask=empir_onehot, bayes_prob=bayes_p_onehot, bayes_truth=bayes_t_onehot, name=surface)
            plt.show()

    def ROC_ctype(self, seed=2553149187, validation_frac=0.15):
        """
        Produces ROCs of CALIOP cloud types.

        Parameters
        -----------
        seed: int
            the seed used to randomly shuffle the data for that model

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data

        para_num: int
            the number of inputs take by the model

        Returns
        ---------
        Matplotlib plots
        """

        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]
        valdf['FCF_RightShift9'] = pd.Series(valdf['Feature_Classification_Flags'].values >> 9,
                                             index=valdf.index)

        bitmeanings = {
            'Low overcast, transparent': 0,
            'Low overcast, opaque': 1,
            'Transition stratocumulus': 2,
            'Low, broken cumulus': 3,
            'Altocumulus (transparent)': 4,
            'Altostratus (opaque)': 5,
            'Cirrus (transparent)': 6,
            'Deep convective (opaque)': 7}

        # Seperate clear flags
        cleardf = valdf[valdf['Feature_Classification_Flags'] & 7 != 2]

        # Truth
        truth = cleardf['CTruth']
        truth_onehot = np.vstack((truth, ~truth)).T

        # Model
        model_confidence = cleardf['Label_Confidence']
        model_onehot = np.vstack(
            (model_confidence, 1 - model_confidence)).T

        # Bayesian mask
        bayes_labels = cleardf['bayes_in']
        bayes_labels[bayes_labels > 1] = 1
        bayes_onehot = np.vstack((bayes_labels, ~bayes_labels)).T

        # Bayesian prob
        bayes_p = cleardf['BayesProb']
        bayes_p_onehot = np.vstack((bayes_p, 1 - bayes_p)).T

        # Empirical mask
        empir_labels = cleardf['cloud_an']
        empir_labels[empir_labels > 1] = 1
        empir_onehot = np.vstack((empir_labels, ~empir_labels)).T

        print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

        me.ROC(model_onehot, truth_onehot, bayes_mask=bayes_onehot,
               emp_mask=empir_onehot, bayes_prob=bayes_p_onehot, name='Clear')

        # Seperate cloudy flags
        cloudydf = valdf[valdf['Feature_Classification_Flags'] & 7 == 2]

        for cloud in bitmeanings:

            cloud_df = cloudydf[cloudydf['FCF_RightShift9']
                                & 7 == bitmeanings[cloud]]

            # Truth
            truth = cloud_df['CTruth']
            truth_onehot = np.vstack((truth, ~truth)).T

            # Model
            model_confidence = cloud_df['Label_Confidence']
            model_onehot = np.vstack(
                (model_confidence, 1 - model_confidence)).T

            # Bayesian mask
            bayes_labels = cloud_df['bayes_in']
            bayes_labels[bayes_labels > 1] = 1
            bayes_onehot = np.vstack((bayes_labels, ~bayes_labels)).T

            # Bayesian prob
            bayes_p = cleardf['BayesProb']
            bayes_p_onehot = np.vstack((bayes_p, 1 - bayes_p)).T

            # Empirical mask
            empir_labels = cloud_df['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_onehot = np.vstack((empir_labels, ~empir_labels)).T

            print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

            me.ROC(model_onehot, truth_onehot, bayes_mask=bayes_onehot,
                   emp_mask=empir_onehot, bayes_prob=bayes_p_onehot, name=cloud)
            plt.show()

    def ROC_modelsens(self, seed=2553149187, validation_frac=0.15):

        self._model_applied()

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        truth = (valdf['CTruth'].values).astype('int')

        false_positive_rate1, true_positive_rate1, _ = metrics.roc_curve(
            truth, valdf['Label_Confidence'].values, pos_label=1)

        false_positive_rate2, true_positive_rate2, _ = metrics.roc_curve(
            truth, valdf['Shuffled_Confidence'].values, pos_label=1)

        plt.figure('ROC')
        plt.title('Model sensitivity to ' + self.shuffled_channel + ' ROC')
        plt.plot([0, 1], [0, 1], label="Random classifier")
        plt.plot(false_positive_rate1, true_positive_rate1,
                 label='Model on original dataframe')
        plt.plot(false_positive_rate2, true_positive_rate2,
                 label='Model on shuffled dataframe')
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.legend()
        plt.show()

    def reproducibility(self, modelname, number_of_runs=15, validation_frac=0.15, para_num=22):
        """
        Return the average and standard deviation of a same model but different
        order of the data it is presented. These outputs quantify the
        reproducibilty  of the model.

        Parameters
        -----------
        modelname: str
            refers to model architecture to run

        number of runs: int
            number of times to run the model.

        validation_frac: float
            the fraction of data kept for validation when preparing the model's training data.

        para_num: int
            the number of inputs take by the model.

        Returns
        ---------
        average: float,
            average accuracy of the model.

        std: float
            standard deviation of the model.
        """

        accuracies = []

        for i in range(number_of_runs):
            model = FFN('Reproducibilty', modelname, 21)
            tdata, vdata, ttruth, vtruth = self._obj.dp.get_ffn_training_data(
                validation_frac=validation_frac, input_type=para_num)
            model.Train(tdata, ttruth, vdata, vtruth)
            acc = me.get_accuracy(model, vdata, vtruth, para_num=para_num)
            accuracies.append(acc)

        average = np.mean(accuracies)
        std = np.std(accuracies)

        return average, std
