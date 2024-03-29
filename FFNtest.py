
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import datetime

import numpy as np
import pandas   # noqa: F401 # pylint: disable=unused-import # Prevent tflearn importing dodgy version
import tflearn
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression

import DataPreparation as dp


class FFN():
    """Object for handling TFLearn DNN models with added support for saving / loading different network configurations"""

    def __init__(self, name, networkConfig=None, para_num=24, LR=1e-3, neuron_num=32, hidden_layers=4,
                 batch_size=64, epochs=10, dout=0.8):
        self.name = name
        self.networkConfig = networkConfig
        self.para_num = para_num
        self.LR = LR
        self.neuron_num = neuron_num
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dout
        self.isLoaded = False
        self._model = None
        self._network = None
        self.run_id = None

    def __str__(self):
        out = ('Model: ' + self.name + '\n'
               + 'Network type: ' + str(self.networkConfig) + '\n'
               + 'Number of inputs: ' + str(self.para_num))
        return(out)

    def TestNetwork(self):
        # Network layers

        # layer 0: generates a 4D tensor
        layer0 = input_data(shape=[None, self.para_num], name='input')
        dout = dropout(layer0, self.dropout)

        for hl in range(self.hidden_layers):
            hidden_layer = fully_connected(
                dout, self.neuron_num, activation='leakyrelu')
            dout = dropout(hidden_layer, self.dropout)

        # Last layer needs to spit out the number of categories
        # we are looking for.
        softmax = fully_connected(dout, 2, activation='softmax')

        # gives the paramaters to optimise the network
        self._network = regression(softmax, optimizer='Adam', learning_rate=self.LR,
                                   loss='categorical_crossentropy', name='targets')
        self.networkConfig = 'TestNetwork'

    @property
    def network(self):
        if self._network is not None:
            return self._network

        if self.networkConfig is None:
            print('Using default network configuration, Network0')
            self.TestNetwork()
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
            self.network, tensorboard_verbose=0, tensorboard_dir='./Temp/tflearn_logs')

        return self._model

    def Train(self, training_data, training_truth, validation_data, validation_truth):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_id = 'Models/' + str(self.name) + '_' + timestamp
        self.model.fit(training_data, training_truth, n_epoch=self.epochs,
                       validation_set=(validation_data, validation_truth),
                       snapshot_step=10000, show_metric=True, run_id=self.run_id)
        self.isLoaded = True

    def Save(self, path=None):
        if path:
            self.model.save(path)
            with open(path + '.txt', 'w') as file:
                file.write(self.networkConfig + '\n')
                file.write(str(self.para_num) + '\n')
                file.write(str(self.LR) + '\n')
                file.write(str(self.neuron_num) + '\n')
                file.write(str(self.hidden_layers) + '\n')
                file.write(str(self.batch_size) + '\n')
                file.write(str(self.epochs) + '\n')
                file.write(str(self.dropout) + '\n')
                file.write(str(self.run_id))
        else:
            self.model.save("Models/" + self.name)
            with open("Models/" + self.name + '.txt', 'w') as file:
                file.write(self.networkConfig + '\n')
                file.write(str(self.para_num) + '\n')
                file.write(str(self.LR) + '\n')
                file.write(str(self.neuron_num) + '\n')
                file.write(str(self.hidden_layers) + '\n')
                file.write(str(self.batch_size) + '\n')
                file.write(str(self.epochs) + '\n')
                file.write(str(self.dropout) + '\n')
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
                    self.LR = int(settings[2].strip())
                    self.neuron_num = int(settings[3].strip())
                    self.hidden_layers = int(settings[4].strip())
                    self.batch_size = int(settings[5].strip())
                    self.epochs = int(settings[6].strip())
                    self.dropout = int(settings[7].strip())

            self.model.load(path)

        else:
            with open('Models/' + self.name + '.txt', 'r') as file:
                settings = file.readlines()
                if len(settings) == 1:
                    self.networkConfig = settings[0]

                elif len(settings) >= 2:
                    self.networkConfig = settings[0].strip()
                    self.para_num = int(settings[1].strip())
                    self.LR = int(settings[2].strip())
                    self.neuron_num = int(settings[3].strip())
                    self.hidden_layers = int(settings[4].strip())
                    self.batch_size = int(settings[5].strip())
                    self.epochs = int(settings[6].strip())
                    self.dropout = int(settings[7].strip())

            self.model.load('Models/' + self.name)

        self.isLoaded = True
        if verbose:
            print('##############################################')
            print('Loading successful')
            print('Model: ' + self.name)
            print('Network type: ' + self.networkConfig)
            print('Number of inputs: ' + str(self.para_num))
            print('Learning rate: ' + str(self.LR))
            print('Neurons per hidden layer: ' + str(self.neuron_num))
            print('Hidden layers: ' + str(self.hidden_layers))
            print('Batch size: ' + str(self.batch_size))
            print('Epochs: ' + str(self.epochs))
            print('Dropout ' + str(self.dropout))
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
    df = dp.PixelLoader('./SatelliteData/SLSTR/Pixels3')

    tdata, vdata, ttruth, vtruth = df.dp.get_ffn_training_data(21)

    model = FFN('Test', 'Network1', 21)
    model.Train(tdata, ttruth, vdata, vtruth)
    model.Save()
