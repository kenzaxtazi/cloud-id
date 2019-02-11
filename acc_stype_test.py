
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu                                        
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import pandas as pd
import ModelEvaluation as me
import matplotlib.pyplot as plt
import sklearn.utils
import DataPreparation as dp
import tflearn
from FFN import FFN
import numpy as np
from tensorflow import reset_default_graph
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import datetime


# If dataset is not created:
'''
training_data, validation_data, training_truth, validation_truth, bayes_values, emp_values = dp.pkl_prep_data(
    '/home/hep/trz15/Matched_Pixels2/Calipso', bayesian=True, empirical=True)
'''

# prepares data for ffn
training_data, validation_data, training_truth, validation_truth, bayes_values, emp_values, _ = dp.pkl_prep_data(
    '/Users/kenzatazi/Desktop/SatelliteData/SLSTR/Pixels2', bayesian=True, empirical=True)

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
