
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import os

import numpy as np

import DataPreparation as dp
from cnn import CNN
from FFN import FFN


class SuperModel():
    def __init__(self, name, FFN1=None, FFN2=None, CNN=None):
        self.name = name
        self.FFN1 = FFN1
        self.FFN2 = FFN2
        self.CNN = CNN
        self._isLoaded = False

    @property
    def isLoaded(self):
        self._isLoaded = (
            self.FFN1.isLoaded and self.FFN2.isLoaded and self.CNN.isLoaded)
        return(self._isLoaded)

    @isLoaded.setter
    def isLoaded(self, value):
        self._isLoaded = value

    def predict_file(self, Sreference):

        ffninputs = dp.getinputsFFN(
            Sreference, input_type=22)  # include indices

        predictions1 = self.FFN1.Predict(ffninputs)[:, 0]
        labels1 = self.FFN1.model.predict_label(ffninputs)[:, 0]

        # boolean mask of bad predictions
        bad = abs(predictions1 - 0.5) < 0.25
        goodindices = ~np.nonzero(bad)
        badindices = np.nonzero(bad)
        cnninputs = dp.getinputsCNN(Sreference, badindices)
        cnninputs = dp.star_padding(cnninputs)

        predictions2 = self.CNN.model.predict(cnninputs)[:, 0]
        labels2 = self.CNN.model.predict_label(cnninputs)[:, 0]

        finallabels = np.zeros(7200000)
        finallabels[goodindices] = labels1
        finallabels[badindices] = labels2

        finalpredictions = np.zeros(7200000)
        finalpredictions[goodindices] = predictions1
        finalpredictions[badindices] = predictions2

        finallabels.reshape((2400, 3000))
        finalpredictions.reshape((2400, 3000))

        return finallabels, finalpredictions

    def Save(self):
        os.mkdir('Models/' + self.name)
        self.FFN1.Save('Models/' + self.name + '/FFN1_' + self.FFN1.name)
        self.FFN2.Save('Models/' + self.name + '/FFN2_' + self.FFN2.name)
        self.CNN.Save('Models/' + self.name + '/CNN_' + self.CNN.name)
        with open('Models/' + self.name + '/Info.txt') as file:
            file.write('FFN1: ' + self.FFN1.name + '\n')
            file.write('FFN2: ' + self.FFN2.name + '\n')
            file.write('CNN: ' + self.CNN.name)

    def Load(self):
        try:
            os.chdir('Models/' + self.name)
        except FileNotFoundError:
            raise Exception('File does not exist')

        self.FFN1 = FFN('FFN1')
        self.FFN1.Load()

        self.FFN2 = FFN('FFN2')
        self.FFN2.Load()

        self.CNN = CNN('CNN')
        self.CNN.Load()
