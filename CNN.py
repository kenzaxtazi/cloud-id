
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import datetime

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression

import DataPreparation as dp


class CNN():
    """Object for handling TFLearn DNN models with added support for saving / loading different network configurations"""

    def __init__(self, name, networkConfig=None, LR=1e-4, img_length=8, img_width=50):
        self.name = name
        self.networkConfig = networkConfig
        self.LR = LR
        self.isLoaded = False
        self.img_length = img_length
        self.img_width = img_width
        self._model = None
        self._network = None
        self.run_id = None

    def __str__(self):
        out = ('Model: ' + self.name + '\n'
               + 'Network type: ' + str(self.networkConfig) + '\n'
               + 'Context shape: ' + str(self.img_length) + ',' + self.img_width)
        return(out)

    def NetworkA(self):
        convnet = input_data(
            shape=[None, self.img_length, self.img_width, 1], name='input')

        # Layer 1
        convnet = conv_2d(convnet, nb_filter=32,
                          filter_size=5, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=5)

        # Layer 2
        convnet = conv_2d(convnet, nb_filter=64,
                          filter_size=5, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=5)

        # Layer 3
        convnet = conv_2d(convnet, nb_filter=128,
                          filter_size=5, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=5)

        # Layer 4
        convnet = conv_2d(convnet, nb_filter=64,
                          filter_size=5, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=5)

        # Layer 5
        convnet = conv_2d(convnet, nb_filter=32,
                          filter_size=5, activation='relu')
        convnet = max_pool_2d(convnet, kernel_size=5)

        # Layer 6
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, keep_prob=0.8)

        # Layer 7
        softmax = fully_connected(convnet, 2, activation='softmax')

        # gives the paramaters to optimise the network
        self._network = regression(softmax, optimizer='Adam', learning_rate=self.LR,
                                   loss='categorical_crossentropy', name='targets')
        self.networkConfig = 'NetworkA'

    @property
    def network(self):
        if self._network is not None:
            return self._network

        if self.networkConfig is None:
            print('Using default network configuration, Network0')
            self.NetworkA()
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

    def Train(self, training_data, training_truth, validation_data, validation_truth, n_epoch=25):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_id = 'Models/' + str(self.name) + '_' + timestamp
        self.model.fit(training_data, training_truth, n_epoch=n_epoch,
                       validation_set=(validation_data, validation_truth),
                       snapshot_step=10000, show_metric=True, run_id=self.run_id)
        self.isLoaded = True

    def Save(self):
        self.model.save("Models/" + self.name)
        with open("Models/" + self.name + '.txt', 'w') as file:
            file.write(self.networkConfig + '\n')
            file.write(str(self.img_length))
            file.write(str(self.img_width))
            file.write(self.run_id)

    def Load(self, verbose=True):
        if self.isLoaded:
            raise AssertionError(
                'Graph already loaded. Consider loading into new object.')

        with open('Models/' + self.name + '.txt', 'r') as file:
            settings = file.readlines()
            self.networkConfig = settings[0].strip()
            self.img_length = int(settings[1].strip())
            self.img_width = int(settings[2].strip())

        self.model.load('Models/' + self.name)
        self.isLoaded = True
        if verbose:
            print('##############################################')
            print('Loading successful')
            print('Model: ' + self.name)
            print('Network type: ' + self.networkConfig)
            print('Context shape: ' + str(self.img_length) + ',' + str(self.img_width))
            print('##############################################')


if __name__ == '__main__':

    df = dp.PixelLoader('./SatelliteData/SLSTR/Pixels5')

    tdata, vdata, ttruth, vtruth = df.dp.get_cnn_training_data()

    model = CNN('CNN_NetA', 'NetworkA')
    model.Train(tdata, ttruth, vdata, vtruth)
    model.Save()
