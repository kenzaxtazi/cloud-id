
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import datetime


import numpy as np
import pandas  # noqa: F401 # pylint: disable=unused-import # Prevent tflearn importing dodgy version
import tflearn
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge


import DataPreparation as dp


class FFN():
    """Object for handling TFLearn DNN models with added support for saving / loading different network configurations"""

    def __init__(self, name, networkConfig=None, para_num=24, LR=1e-3, weights=None):
        self.name = name
        self.networkConfig = networkConfig
        self.para_num = para_num
        self.LR = LR
        self.weights = weights
        self.isLoaded = False
        self._model = None
        self._network = None
        self.run_id = None

    def __str__(self):
        out = ('Model: ' + self.name + '\n'
               + 'Network type: ' + str(self.networkConfig) + '\n'
               + 'Number of inputs: ' + str(self.para_num))
        return(out)

    def Network1(self):
        network = input_data(shape=[None, self.para_num], name='input')
        network = fully_connected(network, 32, activation='leakyrelu')
        network = fully_connected(network, 32, activation='leakyrelu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 32, activation='leakyrelu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 32, activation='leakyrelu')
        network = dropout(network, 0.8)
        softmax = fully_connected(network, 2, activation='softmax')

        weights = input_data(shape=[None, 2], name='weight')
        weighted_softmax = merge([softmax, weights], 'weighted')

        self._network = regression(weighted_softmax, optimizer='Adam', learning_rate=self.LR,
                                   loss='categorical_crossentropy', name='targets', weights=self.weights)
        self.networkConfig = 'Network1'

    @property
    def network(self):
        if self._network is not None:
            return self._network

        else:
            # Use network function specified by networkConfig
            networkFunc = getattr(self, self.networkConfig)
            networkFunc()
        return self._network

    @property
    def model(self):
        if self._model:
            return self._model

        self._model = tflearn.DNN(
            self.network, tensorboard_verbose=0, tensorboard_dir='./Temp/Planing')

        return self._model

    def Train(self, training_data, training_truth, validation_data, validation_truth, training_weights, validation_weights, n_epoch=1):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_id = 'Models/' + str(self.name) + '_' + timestamp
        self.model.fit([training_data, training_weights], training_truth, n_epoch=n_epoch,
                       validation_set=(
                           [validation_data, validation_weights], validation_truth),
                       snapshot_step=10000, show_metric=True, run_id=self.run_id)
        self.isLoaded = True

    def Save(self, path=None):
        if path:
            self.model.save(path)
            with open(path + '.txt', 'w') as file:
                file.write(self.networkConfig + '\n')
                file.write(str(self.para_num) + '\n')
                file.write(str(self.run_id))
        else:
            self.model.save("Models/" + self.name)
            with open("Models/" + self.name + '.txt', 'w') as file:
                file.write(self.networkConfig + '\n')
                file.write(str(self.para_num) + '\n')
                file.write(str(self.run_id))

    def Load(self, path=None, verbose=True):
        if self.isLoaded:
            raise AssertionError(
                'Graph already loaded. Consider loading into new object.')

        if path:
            with open(path + '.txt', 'r') as file:
                settings = file.readlines()
                if len(settings) == 1:
                    self.networkConfig = settings[0]

                elif len(settings) >= 2:
                    self.networkConfig = settings[0].strip()
                    self.para_num = int(settings[1].strip())

            self.model.load(path)

        else:
            with open('Models/' + self.name + '.txt', 'r') as file:
                settings = file.readlines()
                if len(settings) == 1:
                    self.networkConfig = settings[0]

                elif len(settings) >= 2:
                    self.networkConfig = settings[0].strip()
                    self.para_num = int(settings[1].strip())

            self.model.load('Models/' + self.name)

        self.isLoaded = True
        if verbose:
            print('##############################################')
            print('Loading successful')
            print('Model: ' + self.name)
            print('Network type: ' + self.networkConfig)
            print('Number of inputs: ' + str(self.para_num))
            print('##############################################')

    def Predict(self, X):
        return(self.model.predict(X))

    def Predict_label(self, X):
        return(self.model.predict_label(X))

    def apply_mask(self, Sreference):
        if self.isLoaded is False:
            raise AssertionError(
                'Model is neither loaded nor trained, cannot make predictions')

        inputs = dp.getinputsFFN(Sreference, input_type=self.para_num)

        label = self.model.predict_label(inputs)
        lmask = np.array(label)
        lmask = lmask[:, 0].reshape(2400, 3000)

        prob = self.model.predict(inputs)
        pmask = np.array(prob)
        pmask = pmask[:, 0].reshape(2400, 3000)

        return lmask, pmask


if __name__ == '__main__':
    # Pixel Loading
    # df = pd.read_pickle('./SatelliteData/SLSTR/Pixels3/1804P3.pkl')
    df = dp.PixelLoader('./SatelliteData/SLSTR/Pixels3')

    tdata, vdata, ttruth, vtruth = df.dp.get_ffn_training_data(22)

    # tdata = tdata[:-3]
    # ttruth = ttruth[:-3]
    # vdata = vdata[:421184]
    # vtruth = vtruth[:421184]

    tweights = np.random.random((tdata.shape[0], 1))
    tweights = np.concatenate((tweights, ttruth[:, 0].reshape(-1, 1)), axis=1)
    vweights = np.ones((vdata.shape[0], 1))
    vweights = np.concatenate((vweights, vtruth[:, 0].reshape(-1, 1)), axis=1)

    model = FFN('Test', 'Network1', 22)
    model.Train(tdata, ttruth, vdata, vtruth, tweights, vweights)
    model.Save()
