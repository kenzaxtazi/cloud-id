
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import os
import pandas as pd
import numpy as np
import ModelEvaluation as me
import matplotlib.pyplot as plt
import datetime
import sklearn.utils
import DataPreparation as dp
import tflearn
import tensorflow as tf
from tensorflow import reset_default_graph
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from ffn2 import FFN


'''
training_data, validation_data, training_truth, validation_truth, bayes_values, emp_values = dp.pkl_prep_data(
    '/home/hep/trz15/Matched_Pixels2/Calipso', bayesian=False, empirical=False)
'''

# prepares data for ffn
training_data, validation_data, training_truth, validation_truth, _, _, _ = dp.pkl_prep_data(
    '/Users/kenzatazi/Desktop/SatelliteData/SLSTR/Pixels2')


# If dataset already created :
'''
training_data = np.load('training_data.npy')
validation_data = np.load('validation_data.npy')
training_truth = np.load('training_truth.npy')
validation_truth =np.load('validation_truth.npy')
'''

# If dataset already created :
'''
training_data = np.load('training_data.npy')
validation_data = np.load('validation_data.npy')
training_truth = np.load('training_truth.npy')
validation_truth =np.load('validation_truth.npy')
'''

# Creating network and setting hypermarameters for model

# MACHINE LEARNING MODEL
LR = 1e-3  # learning rate
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
MODEL_NAME = 'Models/ffn_withancillarydata_' + timestamp
para_num = 24

# reshape data to pit into network
training_data = training_data.reshape(-1, para_num)
validation_data = validation_data.reshape(-1, para_num)

model = FFN('Net1_S_FFN', 'Network1')
model.networkSetup()
model.Setup()
model.Train(training_data, training_truth, validation_data,
            validation_truth)


# VALIDATION

angle_slices = np.linspace(3, 55, 18)
accuracies = []
N = []

for a in angle_slices:

    new_validation_data = []
    new_validation_truth = []

    # slices
    for i in range(len(validation_data)):
        if abs(validation_data[i, 9]) > a:
            if abs(validation_data[i, 9]) < a+3:
                new_validation_data.append(validation_data[i])
                new_validation_truth.append(validation_truth[i])

    new_validation_data = np.array(new_validation_data)
    new_validation_truth = np.array(new_validation_truth)

    if len(new_validation_data) > 0:

        new_validation_data = new_validation_data.reshape(-1, para_num)
        acc = me.get_accuracy(
            model.model, new_validation_data, new_validation_truth)
        accuracies.append(acc)
        N.append(len(new_validation_data))

    else:
        accuracies.append(0)
        N.append(0)


plt.figure('Accuracy vs satellite zenith angle')
plt.title('Accuracy as a function of satellite zenith angle')
plt.xlabel('Satellite zenith angle (deg)')
plt.ylabel('Accuracy')
plt.bar(angle_slices, accuracies, width=3, align='edge', color='lavenderblush',
        edgecolor='thistle', yerr=(np.array(accuracies)/np.array(N))**(0.5))
plt.show()

# resets the tensorflow environment
reset_default_graph()
