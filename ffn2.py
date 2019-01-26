#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:37:26 2018

@author: kenzatazi
"""

import datetime
import os

import matplotlib.pyplot as plt
import sklearn.utils
import tflearn
from tensorflow import reset_default_graph
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression

import DataPreparation as dp
import ModelApplication as app
import ModelEvaluation as me
import DataLoader as DL
import Visualisation as Vis


class FFN():
    """Object for handling TFLearn DNN models with added support for saving / loading different network configurations"""

    def __init__(self, name, networkConfig=None, para_num=24, LR=1e-3):
        self.name = name
        self.networkConfig = networkConfig
        self.para_num = para_num
        self.LR = LR

    def networkSetup(self):
        """Setup network for the model. Specify network configuration by setting the networkConfig attribute"""
        if self.networkConfig == None:  # No network configuration specified
            self.Network0()  # Use default network
        else:
            # Use network function specified by networkConfig
            networkFunc = getattr(self, self.networkConfig)
            networkFunc()

    def Network0(self):
        # Networks layers

        # layer 0: generates a 4D tensor
        layer0 = input_data(shape=[None, self.para_num], name='input')

        # layer 1
        layer1 = fully_connected(layer0, 32, activation='relu')
        dropout1 = dropout(layer1, 0.8)

        # layer 2
        layer2 = fully_connected(dropout1, 32, activation='relu')
        dropout2 = dropout(layer2, 0.8)

        # layer 3
        layer3 = fully_connected(dropout2, 32, activation='relu')
        dropout3 = dropout(layer3, 0.8)

        # layer 4
        layer4 = fully_connected(dropout3, 32, activation='relu')
        dropout4 = dropout(layer4, 0.8)

        # layer 5 this layer needs to spit out the number of categories
        # we are looking for.
        softmax = fully_connected(dropout4, 1, activation='softmax')

        # gives the paramaters to optimise the network
        self.network = regression(softmax, optimizer='Adam', learning_rate=self.LR,
                                  loss='categorical_crossentropy', name='targets')
        self.networkConfig = 'Network0'

    def Network1(self):
        # Network layers

        # layer 0: generates a 4D tensor
        layer0 = input_data(shape=[None, self.para_num], name='input')

        # layer 1
        layer1 = fully_connected(layer0, 32, activation='leakyrelu')
        # dropout1 = dropout(layer1, 0.8)

        # layer 2
        layer2 = fully_connected(layer1, 32, activation='leakyrelu')
        dropout2 = dropout(layer2, 0.8)

        # layer 3
        layer3 = fully_connected(dropout2, 32, activation='leakyrelu')
        dropout3 = dropout(layer3, 0.8)

        # layer 4
        layer4 = fully_connected(dropout3, 32, activation='leakyrelu')
        dropout4 = dropout(layer4, 0.8)

        # layer 5 this layer needs to spit out the number of categories
        # we are looking for.
        softmax = fully_connected(dropout4, 2, activation='softmax')

        # gives the paramaters to optimise the network
        self.network = regression(softmax, optimizer='Adam', learning_rate=self.LR,
                                  loss='categorical_crossentropy', name='targets')
        self.networkConfig = 'Network1'

    def Network2(self):
        # Network layers

        # layer 0: generates a 4D tensor
        layer0 = input_data(shape=[None, self.para_num], name='input')

        # layer 1
        layer1 = fully_connected(layer0, 32, activation='linear')
        # dropout1 = dropout(layer1, 0.8)

        # layer 2
        layer2 = fully_connected(layer1, 32, activation='relu')
        dropout2 = dropout(layer2, 0.8)

        # layer 3
        layer3 = fully_connected(dropout2, 32, activation='relu')
        dropout3 = dropout(layer3, 0.8)

        # layer 4
        layer4 = fully_connected(dropout3, 32, activation='relu')
        dropout4 = dropout(layer4, 0.8)

        # layer 5 this layer needs to spit out the number of categories
        # we are looking for.
        softmax = fully_connected(dropout4, 2, activation='softmax')

        # gives the paramaters to optimise the network
        self.network = regression(softmax, optimizer='Adam', learning_rate=self.LR,
                                  loss='categorical_crossentropy', name='targets')
        self.networkConfig = 'Network2'

    def Setup(self):
        self.model = tflearn.DNN(self.network, tensorboard_verbose=0)

    def Train(self, training_data, training_truth, validation_data, validation_truth):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        MODEL_NAME = 'Models/ffn_withancillarydata_' + timestamp
        self.model.fit(training_data, training_truth, n_epoch=2,
                       validation_set=(validation_data, validation_truth),
                       snapshot_step=10000, show_metric=True, run_id=MODEL_NAME)

    def Save(self):
        self.model.save("Models/" + self.name)
        with open("Models/" + self.name + '.txt', 'w') as file:
            file.write(self.networkConfig)

    def Load(self):
        with open('Models/' + self.name + '.txt', 'r') as file:
            self.networkConfig = file.read()
            print(self.networkConfig)
        self.networkSetup()
        self.Setup()
        self.model.load('Models/' + self.name)


if __name__ == '__main__':
    # Pixel Loading

    if os.path.exists('/vols/lhcb/egede/cloud'):
        # Script is running on lx02
        scenes = ['/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20171120T190102_20171120T190402_20171122T003854_0179_024_341_2880_LN2_O_NT_002.SEN3',
                  '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20171217T190102_20171217T190402_20171218T223735_0179_025_341_2879_LN2_O_NT_002.SEN3',
                  '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20180113T190102_20180113T190402_20180114T230219_0179_026_341_2880_LN2_O_NT_002.SEN3',
                  '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20180209T190102_20180209T190402_20180210T234449_0179_027_341_2880_LN2_O_NT_002.SEN3']
        pixel_info = dp.PixelLoader("/home/hep/trz15/Matched_Pixels2/Calipso")

    if os.path.exists('/Users/kenzatazi'):
        # Script is running on Kenza's laptop
        scenes = ['/Users/kenzatazi/Desktop/S3A_SL_1_RBT____20180529T113003_20180529T113303_20180530T154711_0179_031_351_1620_LN2_O_NT_003.SEN3']
        pixel_info = dp.PixelLoader("/Users/kenzatazi/Desktop")

    if os.path.exists('D:'):
        scenes = []
        pixel_info = dp.PixelLoader(r"D:\SatelliteData\SLSTR\Pixels2")

    pixels = sklearn.utils.shuffle(pixel_info)

    pixel_values = (pixels[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an',
                            'S7_in', 'S8_in', 'S9_in', 'satellite_zenith_angle',
                            'solar_zenith_angle', 'latitude_an', 'longitude_an',
                            'confidence_an', 'bayes_in',
                            'Feature_Classification_Flags', 'TimeDiff']]).values

    # If dataset is not created:

    # prepares data for ffn
    training_data, validation_data, training_truth, validation_truth, _ = dp.prep_data(
        pixel_values)

    # If dataset already created :
    '''
    training_data = np.load('training_data.npy')
    validation_data = np.load('validation_data.npy')
    training_truth = np.load('training_truth.npy')
    validation_truth =np.load('validation_truth.npy')
    '''

    # MACHINE LEARNING MODEL

    # Creating network and setting hypermarameters for model

    LR = 1e-3  # learning rate

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    MODEL_NAME = 'Models/ffn_withancillarydata_' + timestamp

    para_num = training_data.shape[-1]

    # reshape data to pit into network
    training_data = training_data.reshape(-1, para_num)
    validation_data = validation_data.reshape(-1, para_num)

    model = FFN('Net2_S_FFN', 'Network2')
    model.networkSetup()
    model.Setup()
    model.Train(training_data, training_truth,
                validation_data, validation_truth)
    model.Save()

    Sfile = r"D:\SatelliteData\SLSTR\Dataset1\S3A_SL_1_RBT____20180822T000619_20180822T000919_20180822T015223_0179_035_016_3240_SVL_O_NR_003.SEN3"

    mask1 = app.apply_mask(model.model, Sfile)[0]

    # bmask = DL.extract_mask(Sfile, 'cloud_an', 64)
    bmask = DL.extract_mask(Sfile, 'bayes_in', 2)

    Vis.MaskComparison(Sfile, mask1, bmask, True, 1000)
