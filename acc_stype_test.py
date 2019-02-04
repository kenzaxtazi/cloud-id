#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:05:53 2019

@author: kenzatazi
"""

import pandas as pd
import ModelEvaluation as me
import matplotlib.pyplot as plt
import sklearn.utils
import DataPreparation as dp
import tflearn
from ffn2 import FFN
import numpy as np
from tensorflow import reset_default_graph
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import datetime

# Training data files on HEP server
'''
pixel_info1 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/Calipso/\
                             Apr18P3.pkl")
pixel_info2 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/Calipso/\
                             May18P3.pkl")
pixel_info3 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/Calipso/\
                             Feb18P3.pkl")
pixel_info4 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/Calipso/\
                             Mar18P3.pkl")
pixel_info5 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/Calipso/\
                             Jun18P3.pkl")
pixel_info6 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/Calipso/\
                             Jul18P3.pkl")
pixel_info7 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/Calipso/\
                             Aug18P3.pkl")
pixel_info8 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/Calipso/\
                             Jan18P3.pkl")
pixel_info9 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/CATS/\
                             Aug17P3.pkl")
pixel_info10 = pd.read_pickle("/home/hep/trz15/Matched_Pixels2/CATS/\
                              May17P3.pkl")
'''

# Training data files on users local laptop
pixel_info1 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/Apr18P3.pkl")
pixel_info2 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/May18P3.pkl")
pixel_info3 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/Feb18P3.pkl")
pixel_info4 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/Mar18P3.pkl")
pixel_info5 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/Jun18P3.pkl")
pixel_info6 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/Jul18P3.pkl")
pixel_info7 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/Aug18P3.pkl")
pixel_info8 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/Jan18P3.pkl")
pixel_info9 = pd.read_pickle(
    "/Users/kenzatazi/Desktop/SatelliteData/Nov18P3.pkl")


pixel_info = pd.concat([pixel_info1, pixel_info2, pixel_info3, pixel_info4,
                        pixel_info5, pixel_info6, pixel_info7, pixel_info8, pixel_info9],
                       sort=False)

# normal call
"""
pixel_values = (pixel_info[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an',
                        'S7_in', 'S8_in', 'S9_in',
                        'satellite_zenith_angle', 'solar_zenith_angle',
                        'latitude_an', 'longitude_an',
                        'confidence_an', 'bayes_in', 'cloud_an',
                        'Feature_Classification_Flags',
                        'TimeDiff']]).values
"""

# to call bayesian values for ROC curve
pixel_values = (pixel_info[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an',
                            'S7_in', 'S8_in', 'S9_in', 'satellite_zenith_angle',
                            'solar_zenith_angle', 'latitude_an', 'longitude_an',
                            'confidence_an', 'bayes_in', 'cloud_an',
                            'Feature_Classification_Flags', 'TimeDiff']]).values

# If dataset is not created:

# prepares data for ffn
training_data, validation_data, training_truth, validation_truth, bayes_values, emp_values = dp.pkl_prep_data(
    '/Users/kenzatazi/Desktop/SatelliteData', bayesian=True, empirical=True)

surftype_list = dp.surftype_class(
    validation_data, validation_truth, masks=np.column_stack(
        (bayes_values, emp_values)))

# If dataset already created :

'''
training_data = np.load('training_data.npy')
validation_data = np.load('validation_data.npy')
training_truth = np.load('training_truth.npy')
validation_truth =np.load('validation_truth.npy')
'''

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


accuracies = []
N = []

names = ['Coastline', 'Ocean', 'Tidal', 'Land', 'Inland water',
         'Cosmetic', 'Duplicate', 'Day', 'Twilight', 'Sun glint', 'Snow']

for i in range(len(surftype_list)):

    if len(surftype_list[i]) > 0:
        a = np.concatenate(surftype_list[i])
        b = a.reshape(-1, 3)

        acc = me.get_accuracy(model.model, b[:, 0], b[:, 1])
        labels = (np.concatenate(b[:, 1])).reshape((-1, 2))
        masks = (np.concatenate(b[:, 2])).reshape((-1, 2))

        bayes_mask = masks[:, 0]
        emp_mask = masks[:, 1]
        bayes_mask[bayes_mask > 1.0] = 1.0
        emp_mask[emp_mask > 1.0] = 1.0

        bayes_acc = 1 - np.mean(np.abs(labels[:, 0] - bayes_mask))
        emp_acc = 1 - np.mean(np.abs(labels[:, 0] - emp_mask))
        me.ROC_curve(model.model, b[:, 0], b[:, 1],
                     bayes_mask=bayes_mask, emp_mask=emp_mask, name=names[i])
        accuracies.append([acc, bayes_acc, emp_acc])
        N.append(len(surftype_list[i]))

    else:
        accuracies.append([0, 0, 0])
        N.append(0)

accuracies = (np.concatenate(np.array(accuracies))).reshape(-1, 3)

t = np.arange(len(names))

plt.figure('Accuracy vs surface type')
plt.title('Accuracy as a function of surface type')
plt.ylabel('Accuracy')
bars = plt.bar(t, accuracies[:, 0], width=0.5, align='center', color='honeydew',
               edgecolor='palegreen',  yerr=(np.array(accuracies[:, 0])/np.array(N))**(0.5),
               tick_label=names, zorder=1)
circles = plt.scatter(t, accuracies[:, 1], marker='o', zorder=2)
stars = plt.scatter(t, accuracies[:, 2], marker='*', zorder=3)
plt.xticks(rotation=90)
plt.legend([bars, circles, stars], ['Model accuracy',
                                    'Bayesian mask accuracy', 'Empirical mask accuracy'])
plt.show()

# resets the tensorflow environment
reset_default_graph()
