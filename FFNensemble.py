
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import datetime

import numpy as np
import pandas   # noqa: F401 # pylint: disable=unused-import # Prevent tflearn importing dodgy version
import tflearn
import tensorflow as tf 
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression

import DataPreparation as dp
from FFN import FFN


class FFNensemble():
    """Object for handling TFLearn DNN models with added support for saving / loading different network configurations"""

    def __init__(self, name, networkConfig, ensemble_num=10, para_num=21):
        self.name = name
        self.ensemble_num = ensemble_num
        self.networkConfig = networkConfig
        self.para_num = para_num

    def Train_and_Save(self, training_data, training_truth, validation_data, validation_truth, epoch_num=16):

        for n in range(self.ensemble_num):

            model = FFN(str(n), self.networkConfig, self.para_num)
            model.Train(training_data, training_truth, validation_data, validation_truth, epochs=epoch_num)
            model.Save("Models/" + self.name + "/" + str(n))
            tf.reset_default_graph()
    
    def Predict(self, X):
        Y = []
        for n in range(self.ensemble_num):
            model = FFN("Models/" + self.name + "/" + str(n))
            model.Load()
            y = model.Predict(X)
            Y.append(y)
            Y = np.array(Y)
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            tf.reset_default_graph()

        return(Y_mean, Y_std)

    def apply_mask(self, Sreference):
        inputs = dp.getinputsFFN(Sreference, input_type=self.para_num)
        prob, var = self.Predict(inputs)

        pmask = np.array(prob)
        vmask = np.array(var)

        pmask = pmask[:, 0].reshape(2400, 3000)
        vmask = vmask[:, 0].reshape(2400, 3000)

        return pmask, vmask


if __name__ == '__main__':
    # Pixel Loading

    df = dp.PixelLoader('./SatelliteData/SLSTR/Pixels3')

    tdata, vdata, ttruth, vtruth = df.dp.get_ffn_training_data(21)

    ensemble = FFNensemble('Test', 'Network1', 21)
    ensemble.Train_and_Save(tdata, ttruth, vdata, vtruth)
