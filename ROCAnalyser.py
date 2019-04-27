##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import ModelEvaluation as me
import seaborn as sns
from FFN import FFN


@pd.api.extensions.register_dataframe_accessor("roc")
class ROCAnalyser():
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.model = None
        self.shuffled_channel = None

    def _model_applied(self):
        """Raise error if Agree column is not in dataframe"""
        if 'Agree' not in self._obj.columns:
            raise AttributeError(
                'No model has been applied to this dataframe.'
                ' See df.roc.model_agreement')

    def model_agreement(self, model, verbose=False, MaxDist=None, MaxTime=None):
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
            model = FFN(model)
            model.Load(verbose=verbose)

        elif isinstance(model, FFN):
            pass

        num_inputs = model.para_num
        inputs = self._obj.dp.get_ffn_inputs(num_inputs)
        output_labels = model.model.predict_label(inputs)
        output_con = model.model.predict(inputs)

        self._obj['Labels'] = pd.Series(
            output_labels[:, 0], index=self._obj.index)
        self._obj['Label_Confidence'] = pd.Series(
            output_con[:, 0], index=self._obj.index)

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

    def average(self, seed=2553149187, validation_frac=0.15):
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

        self._obj.dp.remove_nan()
        self._obj.dp.remove_anomalous()
        self._obj.dp.shuffle_by_file(seed)

        self._obj = self._obj.dp._obj   # Assign the filtered dataframe to self._obj

        pct = int(len(self._obj) * validation_frac)
        valdf = self._obj[-pct:]

        # Truth
        truth = valdf['CTruth']
        truth_onehot = np.vstack((truth, ~truth)).T

        # Model
        model_confidence = valdf['Label_Confidence']
        model_onehot = np.vstack(
            (model_confidence, 1 - model_confidence)).T

        # Bayesian mask
        bayes_labels = valdf['bayes_in']
        bayes_labels[bayes_labels > 1] = 1
        bayes_onehot = np.vstack((bayes_labels, ~bayes_labels)).T

        # Empirical mask
        empir_labels = valdf['cloud_an']
        empir_labels[empir_labels > 1] = 1
        empir_onehot = np.vstack((empir_labels, ~empir_labels)).T

        me.ROC(model_onehot, truth_onehot, bayes_mask=bayes_onehot,
               emp_mask=empir_onehot)

        plt.show()

    def stype(self, seed=2553149187, validation_frac=0.15):
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
            'NDSI snow': 8192,
            'Cosmetic': 256,
            'Duplicate': 512,
            'Day': 1024,
            'Twilight': 2048
            }
        
        validation_predictions = []
        validation_truths = []
        bayes_masks = []
        emp_masks = []
        names = []

        for surface in bitmeanings:

            if surface != 'Dry Land':
                surfdf = valdf[valdf['confidence_an']
                               & bitmeanings[surface] == bitmeanings[surface]]
            else:
                surfdf = valdf[valdf['confidence_an']
                               & bitmeanings[surface] == 8]

            names.append(surface)

            # Truth
            truth = surfdf['CTruth']
            truth_onehot = np.vstack((truth, ~truth)).T
            validation_truths.append(truth_onehot)

            # Model
            model_confidence = surfdf['Label_Confidence']
            model_onehot = np.vstack(
                (model_confidence, 1 - model_confidence)).T
            validation_predictions.append(model_onehot)

            # Bayesian mask
            bayes_labels = surfdf['bayes_in']
            bayes_labels[bayes_labels > 1] = 1
            bayes_onehot = np.vstack((bayes_labels, ~bayes_labels)).T
            bayes_masks.append(bayes_onehot)

            # Empirical mask
            empir_labels = surfdf['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_onehot = np.vstack((empir_labels, ~empir_labels)).T
            emp_masks.append(empir_onehot)
            # print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

            me.ROC(model_onehot, truth_onehot, bayes_mask=bayes_onehot,
                   emp_mask=empir_onehot, name=surface)

        me.nROC(validation_predictions, validation_truths,
                RGBtoHEX(sns.color_palette("husl", 10)),
                bayes_masks, emp_masks, names, title='Surface types')
                
        plt.show()

    def ctype(self, seed=2553149187, validation_frac=0.15):
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
            'low overcast, opaque': 1,      # opaque clouds
            'deep convective (opaque)': 7,
            'altostratus (opaque)': 5,
            
            'altocumulus (transparent)': 4,  # transparent clouds
            'cirrus (transparent)': 6,
            'low overcast, transparent': 0,
            
            'transition stratocumulus': 2,  # broken clouds
            'low, broken cumulus': 3}

        validation_predictions = []
        validation_truths = []
        bayes_masks = []
        emp_masks = []
        names = []
        AUCS =[]

        # Seperate clear flags
        cleardf = valdf[valdf['Feature_Classification_Flags'] & 7 != 2]

        # Truth
        clear_truth = cleardf['CTruth']
        clear_truth_onehot = np.vstack((clear_truth, ~clear_truth)).T

        # Model
        clear_model_confidence = cleardf['Label_Confidence']
        clear_model_onehot = np.vstack(
            (clear_model_confidence, 1 - clear_model_confidence)).T

        # Bayesian mask
        clear_bayes_labels = cleardf['bayes_in']
        clear_bayes_labels[clear_bayes_labels > 1] = 1
        clear_bayes_onehot = np.vstack(
            (clear_bayes_labels, ~clear_bayes_labels)).T

        # Empirical mask
        clear_empir_labels = cleardf['cloud_an']
        clear_empir_labels[clear_empir_labels > 1] = 1
        clear_empir_onehot = np.vstack(
            (clear_empir_labels, ~clear_empir_labels)).T

        # print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

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

            # Empirical mask
            empir_labels = cloud_df['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_onehot = np.vstack((empir_labels, ~empir_labels)).T

            # print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

            combined_model_onehot = np.concatenate(
                (clear_model_onehot, model_onehot))
            validation_predictions.append(combined_model_onehot)

            combined_truth_onehot = np.concatenate(
                (clear_truth_onehot, truth_onehot))
            validation_truths.append(combined_truth_onehot)

            combined_bayes_onehot = np.concatenate(
                (clear_bayes_onehot, bayes_onehot))
            bayes_masks.append(combined_bayes_onehot)

            combined_empir_onehot = np.concatenate(
                (clear_empir_onehot, empir_onehot))
            emp_masks.append(combined_empir_onehot)

            names.append(cloud)

            AUCS.append(metrics.roc_auc_score(combined_truth_onehot[:, 1], combined_model_onehot[:, 1]))

            me.ROC(combined_model_onehot, combined_truth_onehot,
                   bayes_mask=combined_bayes_onehot, emp_mask=combined_empir_onehot,
                   name=cloud)

        me.nROC(validation_predictions[:3], validation_truths[:3],
                RGBtoHEX(sns.cubehelix_palette(3, dark=0.3)),
                bayes_masks[:3], emp_masks[:3], names[:3],
                title='Opaque cloud types')
        
        me.nROC(validation_predictions[3:6], validation_truths[3:6],
                RGBtoHEX(sns.cubehelix_palette(3, rot=-.4, dark=0.3)),
                bayes_masks[3:6], emp_masks[3:6], names[3:6],
                title='Transparent cloud types')

        me.nROC(validation_predictions[-2:], validation_truths[-2:],
                RGBtoHEX(sns.cubehelix_palette(2, rot=-.25, light=.7, dark=0.5)),
                bayes_masks[-2:], emp_masks[-2:], names[-2:], title='Broken cloud types')

        plt.show()

        return AUCS

    def model_sens(self, seed=2553149187, validation_frac=0.15):

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

    def arctic_antarctic(self, seed=2553149187, validation_frac=0.15):
        """
        Produces ROCs of the Arctic and Antarctic validation data.

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

        seperated_valdf = [valdf[valdf['Latitude'] > 0],
                           valdf[valdf['Latitude'] < 0]]
        names = ['Arctic', 'Antarctic']

        validation_predictions = []
        validation_truths = []
        bayes_masks = []
        emp_masks = []
        AUCS = []

        for pole in seperated_valdf:

            # Truth
            truth = pole['CTruth']
            truth_onehot = np.vstack((truth, ~truth)).T
            validation_truths.append(truth_onehot)

            # Model
            model_confidence = pole['Label_Confidence']
            model_onehot = np.vstack(
                (model_confidence, 1 - model_confidence)).T
            validation_predictions.append(model_onehot)

            # Bayesian mask
            bayes_labels = pole['bayes_in']
            bayes_labels[bayes_labels > 1] = 1
            bayes_onehot = np.vstack((bayes_labels, ~bayes_labels)).T
            bayes_masks.append(bayes_onehot)

            # Empirical mask
            empir_labels = pole['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_onehot = np.vstack((empir_labels, ~empir_labels)).T
            emp_masks.append(empir_onehot)
            # print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

            AUCS.append(metrics.roc_auc_score(truth_onehot[:, 1], model_onehot[:, 1])) 

        me.nROC(validation_predictions, validation_truths,
                RGBtoHEX(sns.cubehelix_palette(2, start=2.8, rot=.1, dark=0.5)),
                bayes_masks, emp_masks, names, title='Arctic and Antarctic')
        plt.show()

        return AUCS

    def land_ocean(self, seed=2553149187, validation_frac=0.15):
        """
        Produces ROCs of the dry land and ocean validation data.

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

        bitmeanings = {
            'ocean': 2,
            'dry land': 24}

        validation_predictions = []
        validation_truths = []
        bayes_masks = []
        emp_masks = []
        names = []
        AUCS = []

        for surface in bitmeanings:

            if surface != 'dry land':
                surfdf = valdf[valdf['confidence_an']
                               & bitmeanings[surface] == bitmeanings[surface]]
            else:
                surfdf = valdf[valdf['confidence_an']
                               & bitmeanings[surface] == 8]

            # Truth
            truth = surfdf['CTruth']
            truth_onehot = np.vstack((truth, ~truth)).T
            validation_truths.append(truth_onehot)

            # Model
            model_confidence = surfdf['Label_Confidence']
            model_onehot = np.vstack(
                (model_confidence, 1 - model_confidence)).T
            validation_predictions.append(model_onehot)

            # Bayesian mask
            bayes_labels = surfdf['bayes_in']
            bayes_labels[bayes_labels > 1] = 1
            bayes_onehot = np.vstack((bayes_labels, ~bayes_labels)).T
            bayes_masks.append(bayes_onehot)

            # Empirical mask
            empir_labels = surfdf['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_onehot = np.vstack((empir_labels, ~empir_labels)).T
            emp_masks.append(empir_onehot)
            # print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

            AUCS.append(metrics.roc_auc_score(truth_onehot[:, 1], model_onehot[:, 1]))      
            names.append(surface)

        me.nROC(validation_predictions, validation_truths, RGBtoHEX(sns.color_palette("husl", 2)),
                bayes_masks, emp_masks, names, title='Land and Ocean')
        plt.show()

        return AUCS


def RGBtoHEX(array):
    """ Turns RGB tuples into HEX strings"""
    hexs = []
    for rgb in array:
        rgb = np.array(rgb) * 255
        hex_str = ('#%02X%02X%02X' % (int(rgb[0]), int(rgb[1]), int(rgb[2])))
        hexs.append(hex_str)
    return hexs
