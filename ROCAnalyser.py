##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ModelEvaluation as me

matplotlib.rcParams.update({'errorbar.capsize': 0.15})


@pd.api.extensions.register_dataframe_accessor("roca")
class ROCAnalysis():

    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.model = None

    def ROC_ctype(self, seed=1, validation_frac=0.15):
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

        self._obj.da._model_applied()

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

        # Empirical mask
        empir_labels = cleardf['cloud_an']
        empir_labels[empir_labels > 1] = 1
        empir_onehot = np.vstack((empir_labels, ~empir_labels)).T

        print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

        me.ROC(model_onehot, truth_onehot, bayes_mask=bayes_onehot,
               emp_mask=empir_onehot, name='Clear')

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

            print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

            me.ROC(model_onehot, truth_onehot, bayes_mask=bayes_onehot,
                   emp_mask=empir_onehot, name=cloud)
            plt.show()

    def ROC_stype(self, seed=1, validation_frac=0.15):
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
        self._obj.da._model_applied()

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

            if surface != 'dry_land':
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

            # Empirical mask
            empir_labels = surfdf['cloud_an']
            empir_labels[empir_labels > 1] = 1
            empir_onehot = np.vstack((empir_labels, ~empir_labels)).T

            print(model_onehot, truth_onehot, bayes_onehot, empir_onehot)

            me.ROC(model_onehot, truth_onehot, bayes_mask=bayes_onehot,
                   emp_mask=empir_onehot, name=surface)
            plt.show()


def ROC_channel_senstitivity(df, channel_name, seed=1, validation_frac=0.15):

    """
    Produces ROC of comparing normal model and shuffled sensitivity.

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

    df.dp.remove_nan()
    df.dp.remove_anomalous()
    df.dp.shuffle_by_file(seed)

    pct = int(len(df) * validation_frac)
    valdf1 = df[-pct:]
    valdf2 = df[-pct:]

    valdf2[channel_name] = pd.Series(np.random.shuffle(valdf2[channel_name].values),
                                     index=valdf2.index)

    valdf1.da.model_agreement('Net1_FFN_v7')
    valdf2.da.model_agreement('Net1_FFN_v7')

    # Truth
    truth = valdf1['CTruth']
    truth_onehot = np.vstack((truth, ~truth)).T

    # No shuffle
    model_confidence1 = valdf1['Label_Confidence']
    model_onehot1 = np.vstack(
        (model_confidence1, 1 - model_confidence1)).T

    # With shuffle
    model_confidence2 = valdf2['Label_Confidence']
    model_onehot2 = np.vstack(
        (model_confidence1, 1 - model_confidence2)).T

    false_positive_rate1, true_positive_rate1, _ = metrics.roc_curve(
        truth_onehot[:, 0], model_onehot1[:, 0], pos_label=1)

    false_positive_rate2, true_positive_rate2, _ = metrics.roc_curve(
        truth_onehot[:, 0], model_onehot2[:, 0], pos_label=1)

    plt.figure(channel_name + ' ' + 'ROC comparison')
    plt.title('ROC comparison')
    plt.plot(false_positive_rate1, true_positive_rate1, label='Original validation data')
    plt.plot(false_positive_rate2, true_positive_rate2, label='Validation data with shuffle channel')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], label="Random classifier")
