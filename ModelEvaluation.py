#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:37:40 2018

@author: kenzatazi
"""

# Evaluating the model

import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
import tflearn
import numpy as np


def get_accuracy(model, validation_data, validation_truth):

    """ returns model accuracy """

    accuracy = model.evaluate(validation_data, validation_truth)
    return accuracy[0]


def ROC_curve(model, validation_data, validation_truth):

    """Plots Receiver Operating Characteristic (ROC) curve"""

    predictions = model.predict(validation_data)

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
            validation_truth[:, 1], predictions[:, 1], pos_label=1)

    plt.figure('ROC curve')
    plt.title('ROC curve')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], label="random classifier")


def AUC(model, validation_data, validation_truth):

    """Returs area under Receiver Operating Characteristic (ROC) curve"""

    predictions = model.predict(validation_data)

    auc = metrics.roc_auc_score(validation_truth[:, 1], predictions[:, 1])
    return auc


def precision_vs_recall(model, validation_data, validation_truth):

    """Plots precision vs recall curve"""

    predictions = np.nan_to_num(model.predict(validation_data))

    precision, recall, thresholds = metrics.precision_recall_curve(
            validation_truth[:, 1], predictions[:, 1], pos_label=1)

    plt.figure('Precision vs recall curve')
    plt.title('Precision vs recall curve')
    plt.plot(precision, recall)
    plt.xlabel('Precision')
    plt.ylabel('Recall')


def confusion_matrix(model, validation_data, validation_truth):

    """ Returns a confusion matrix"""

    labels = model.predict_label(validation_data)
    confusion_matrix = tf.confusion_matrix(validation_truth[:, 1],
                                           labels[:, 1])
    with tf.Session().as_default() as sess:
        m = tf.Tensor.eval(confusion_matrix, feed_dict=None, session=sess)
    return m
