
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import os

import numpy as np

import DataPreparation as dp
from CNN import CNN
from FFN import FFN


class SuperModel():
    def __init__(self, name, FFN=None, CNN=None):
        self.name = name
        self.FFN = FFN
        self.CNN = CNN
        self._isLoaded = False

    @property
    def isLoaded(self):
        self._isLoaded = (
            self.FFN.isLoaded and self.CNN.isLoaded)
        return(self._isLoaded)

    @isLoaded.setter
    def isLoaded(self, value):
        self._isLoaded = value

    def predict_file(self, Sreference):

        ffninputs = dp.getinputsFFN(
            Sreference, input_type=22)  # include indices

        predictions1 = self.FFN.Predict(ffninputs)[:, 0]
        labels1 = self.FFN.model.predict_label(ffninputs)[:, 0]

        # boolean mask of bad predictions
        bad = abs(predictions1 - 0.5) < 0.25
        goodindices = np.where(bad == False)[0]
        badindices = np.where(bad == True)[0]
        cnninputs = dp.getinputsCNN(Sreference, badindices)
        cnninputs = dp.star_padding(cnninputs)

        # Feeding all of the inputs at once can cause a memory error
        # Instead split into chunks of 10,000

        chunkedcnninputs = [cnninputs[i: i + 10000]
                            for i in range(0, len(cnninputs), 10000)]

        predictions2 = []
        labels2 = []

        for i in range(len(chunkedcnninputs)):
            predictions2.extend(self.CNN.model.predict(
                chunkedcnninputs[i])[:, 0])
            labels2.extend(self.CNN.model.predict_label(
                chunkedcnninputs[i])[:, 0])

        finallabels = np.zeros(7200000)
        finallabels[goodindices] = labels1[goodindices]
        finallabels[badindices] = labels2

        finalpredictions = np.zeros(7200000)
        finalpredictions[goodindices] = predictions1[goodindices]
        finalpredictions[badindices] = predictions2

        finallabels = finallabels.reshape((2400, 3000))
        finalpredictions = finalpredictions.reshape((2400, 3000))

        return finallabels, finalpredictions

    def Save(self):
        os.mkdir('Models/' + self.name)
        self.FFN.Save('Models/' + self.name + '/FFN_' + self.FFN.name)
        self.CNN.Save('Models/' + self.name + '/CNN_' + self.CNN.name)
        with open('Models/' + self.name + '/Info.txt') as file:
            file.write('FFN: ' + self.FFN.name + '\n')
            file.write('CNN: ' + self.CNN.name)

    def Load(self):
        try:
            with open('Models/' + self.name + '/Info.txt') as file:
                settings = file.readlines()
                if len(settings) == 2:
                    self.FFN.name = settings[0].strip().split(' ')[1]
                    self.CNN.name = settings[1].strip().split(' ')[1]
        except FileNotFoundError:
            raise Exception('File does not exist')

        self.FFN = FFN(self.FFN.name)
        self.FFN.Load('Models/' + self.name + '/FFN_' + self.FFN.name)

        self.CNN = CNN(self.CNN.name)
        self.CNN.Load('Models/' + self.name + '/CNN_' + self.CNN.name)
