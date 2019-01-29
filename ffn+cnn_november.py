#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:37:26 2018

@author: kenzatazi
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.utils
import tflearn
from satpy import Scene
from tensorflow import reset_default_graph
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

import DataPreparation as dp
import ModelApplication as app
import ModelEvaluation as me
import PixelAnalysis as PA
import datetime


# DATA PREPARATION

# Pixel Loading

if os.path.exists('/vols/lhcb/egede/cloud'):
    # Script is running on lx02
    scenes = ['/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20171120T190102_20171120T190402_20171122T003854_0179_024_341_2880_LN2_O_NT_002.SEN3',
              '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20171217T190102_20171217T190402_20171218T223735_0179_025_341_2879_LN2_O_NT_002.SEN3',
              '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20180113T190102_20180113T190402_20180114T230219_0179_026_341_2880_LN2_O_NT_002.SEN3',
              '/vols/lhcb/egede/cloud/SLSTR/pacific_day/S3A_SL_1_RBT____20180209T190102_20180209T190402_20180210T234449_0179_027_341_2880_LN2_O_NT_002.SEN3']
    pixel_info = PA.PixelLoader("/home/hep/trz15/Matched_Pixels2/Calipso/Nov18P3.pkl")

if os.path.exists('/Users/kenzatazi'):
    # Script is running on Kenza's laptop
    scenes = ['/Users/kenzatazi/Desktop/S3A_SL_1_RBT____20180529T113003_20180529T113303_20180530T154711_0179_031_351_1620_LN2_O_NT_003.SEN3']
    pixel_info = PA.PixelLoader("/Users/kenzatazi/Desktop/Nov18P3.pkl")


pixel_values = (pixels[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an',
                        'S7_in', 'S8_in', 'S9_in', 'satellite_zenith_angle',
                        'solar_zenith_angle', 'latitude_an', 'longitude_an',
                        'confidence_an', 'bayes_in', 'Sfilename',
                        'Feature_Classification_Flags', 'TimeDiff']]).values


# If dataset is not created:

# prepares data for ffn
training_data, validation_data, training_truth, validation_truth,\
    training_images, validation_images = dp.prep_data(pixel_values)


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

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
MODEL_NAME = 'ffn_withancillarydata_' + timestamp
img_size = 300
para_num = len(pixel_values[0, :-2])

# reshape data to pit into network
training_data = training_data.reshape(-1, 1, para_num, 1)
validation_data = validation_data.reshape(-1, 1, para_num, 1)
training_images = training_images.reshape(-1, img_size, img_size, 1)
validation_images = training_images.reshape(-1, img_size, img_size, 1)


# Convolutional Network layers

# Layer 0: generates a 4D tensor
convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

# Layer 1
convnet = conv_2d(convnet, nb_filter=32, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 2
convnet = conv_2d(convnet, nb_filter=64, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 3
convnet = conv_2d(convnet, nb_filter=128, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 4
convnet = conv_2d(convnet, nb_filter=64, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 5
convnet = conv_2d(convnet, nb_filter=32, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 6
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, keep_prob=0.8)

# Layer 7
convnet = fully_connected(convnet, n_units=1, activation='softmax')


# Merge output from CNN with input for FFN

# layer -1: generates a 4D tensor
input_tensor = input_data(shape=[None, 1, para_num, 1], name='input')

layer0 = merge([convnet, input_tensor], mode='concat', axis=1, name='Merge')

# Feed Forward Networks layers

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

# layer 5
layer5 = fully_connected(dropout4, 32, activation='relu')
dropout5 = dropout(layer5, 0.8)

# layer 6 this layer needs to spit out the number of categories
# we are looking for.
softmax = fully_connected(dropout5, 2, activation='softmax')

# gives the paramaters to optimise the network
network = regression(softmax, optimizer='Adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')
# creates the model
model = tflearn.DNN(network, tensorboard_verbose=0)
model.save(MODEL_NAME)

# If model is already created
"""
### UNPACK SAVED DATA
if os.path.exists('/Users/kenzatazi/Documents/University/\
Year 4/Msci Project/{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
"""

# Model training
model.fit(training_data, training_truth, n_epoch=2,
          validation_set=(validation_data, validation_truth),
          snapshot_step=10000, show_metric=True, run_id=MODEL_NAME)


# Print accuracy
acc = me.get_accuracy(model, validation_data, validation_truth)

# apply model to test images to generate masks

'''
for scn in scenes:
    app.apply_mask(model, scn)
'''

# resets the tensorflow environment
reset_default_graph()


plt.show()
