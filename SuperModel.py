
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

        ftestdata = dp.getinputs(Sreference, input_type=24)  # include indices
        indices = np.arange(7200000)

        # exclude indices when predicting
        predictions1 = self.FFN1.Predict(ftestdata)
        augm_data = np.column_stack(
            (predictions1, indices))  # merge with indices

        gooddata = [x for x in augm_data if not(0.4 <= x[0] < 0.6)]
        poordata = [x for x in augm_data if (0.4 <= x[0] < 0.6)]

        ctestdata = dp.cnn_getinputs(Sreference, poordata)
        predictions2 = self.CNN.predict(
            ctestdata[0])   # May require .model.predict

        predictions3 = self.FFN2.predict(
            np.column_stack((augm_data[:, 0], predictions2)))

        fixeddata = np.column_stack((predictions3, poordata[:, -1]))

        final_predictions_with_indices = np.concatenate(gooddata, fixeddata)
        final_predictions = (np.sort(final_predictions_with_indices))[:, 0]

        return final_predictions

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
