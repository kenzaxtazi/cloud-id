
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu                                        
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import datetime
import os

#CNNimport matplotlib.pyplot as plt
import sklearn.utils
import tflearn
from tensorflow import reset_default_graph
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d

import DataPreparation as dp
import ModelApplication as app
import ModelEvaluation as me
import DataLoader as DL
#import Visualisation as Vis


class CNN():
    """Object for handling TFLearn DNN models with added support for saving / loading different network configurations"""

    def __init__(self, name, networkConfig=None, LR=1e-3, img_length=8, img_width=50):
        self.name = name
        self.networkConfig = networkConfig
        self.LR = LR
        self.isLoaded = False
        self.img_length = img_length
        self.img_width = img_width

    def networkSetup(self):
        """Setup network for the model. Specify network configuration by setting the networkConfig attribute"""
        if self.networkConfig == None:  # No network configuration specified
            self.Network0()  # Use default network
        else:
            # Use network function specified by networkConfig
            networkFunc = getattr(self, self.networkConfig)
            networkFunc()

    def Network0(self):
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
        self.network = regression(softmax, optimizer='Adam', learning_rate=self.LR,
                                  loss='categorical_crossentropy', name='targets')
        self.networkConfig = 'cnn'

    def Setup(self):
        self.model = tflearn.DNN(
            self.network, tensorboard_verbose=0, tensorboard_dir='./Temp/tflearn_logs')

    def Train(self, training_data, training_truth, validation_data, validation_truth):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        MODEL_NAME = 'Models/CNN' + timestamp
        self.model.fit(training_data, training_truth, n_epoch=1,
                       validation_set=(validation_data, validation_truth),
                       snapshot_step=10000, show_metric=True, run_id=MODEL_NAME)
        self.isLoaded = True

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
        self.isLoaded = True


if __name__ == '__main__':
    # Pixel Loading

    if os.path.exists('/vols/lhcb/egede/cloud'):
        # Script is running on lx02
        scenes = ['/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20171120T190102_20171120T190402_20171122T003854_0179_024_341_2880_LN2_O_NT_002.SEN3',
                  '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20171217T190102_20171217T190402_20171218T223735_0179_025_341_2879_LN2_O_NT_002.SEN3',
                  '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20180113T190102_20180113T190402_20180114T230219_0179_026_341_2880_LN2_O_NT_002.SEN3',
                  '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20180209T190102_20180209T190402_20180210T234449_0179_027_341_2880_LN2_O_NT_002.SEN3']

    if os.path.exists('/Users/kenzatazi'):
        # Script is running on Kenza's laptop
        scenes = ['/Users/kenzatazi/Desktop/SatelliteData/S3A_SL_1_RBT____20180529T113003_20180529T113303_20180530T154711_0179_031_351_1620_LN2_O_NT_003.SEN3']

    if os.path.exists('D:'):
        scenes = []

    # one month at a time
    training_data, validation_data, training_truth, validation_truth = dp.cnn_prep_data(
        location_directory='/home/hep/trz15/Matched_Pixels2/Calipso/P4',
        context_directory='/home/hep/trz15/Matched_Pixels2/Calipso/Context')

    # MACHINE LEARNING MODEL

    # Creating network and setting hypermarameters for model

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    MODEL_NAME = 'Models/cnn_' + timestamp

    model = CNN('CNN_Net0', 'Network0')
    model.networkSetup()
    model.Setup()
    model.Train(training_data, training_truth,
                validation_data, validation_truth)
    model.Save()

    mask1 = app.apply_mask(model.model, scenes,
                           binary=True, probability=True)[1]

    # bmask = DL.extract_mask(Sfile, 'cloud_an', 64)
    bmask = DL.extract_mask(scenes, 'bayes_in', 2)

    #Vis.MaskComparison(Sfile, mask1, bmask, True, 1000)
    #Vis.simple_mask(mask1, S1)
