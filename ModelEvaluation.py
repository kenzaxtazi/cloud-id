
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


def get_accuracy(model, validation_data, validation_truth, para_num=22):
    """ returns model accuracy """

    validation_data = np.concatenate(validation_data)
    validation_data = validation_data.reshape(-1, para_num)

    validation_truth = np.concatenate(validation_truth)
    validation_truth = validation_truth.reshape(-1, 2)

    accuracy = model.evaluate(validation_data, validation_truth)

    return accuracy[0]


def ROC(validation_predictions, validation_truth, bayes_mask=None,
        emp_mask=None, name=None, validation_predictions2=None,
        validation_truth2=None, bayes_mask2=None, emp_mask2=None,
        name2=None):
    """ Plots Receiver Operating Characteristic (ROC) curve """

    # Set 1

    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(
        validation_truth[:, 0], validation_predictions[:, 0], pos_label=1)

    if name is None:
        if name2 is None:
            plt.figure('ROC')
            plt.title('ROC')
    else:
        plt.figure(name + ' and ' + name2 + ' ROC')
        plt.title(name + ' and ' + name2 + ' ROC')
    plt.plot(false_positive_rate, true_positive_rate, label='Model ' + name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], label="Random classifier")

    if bayes_mask is not None:
        validation_truth = validation_truth.astype(int)
        bayes_mask = bayes_mask.astype(int)
        tn, fp, fn, tp = (metrics.confusion_matrix(
            validation_truth[:, 0], bayes_mask[:, 0], labels=(0, 1))).ravel()
        # print(tn, fp, fn, tp)
        plt.scatter(float(fp) / float(tn + fp), float(tp) /
                    float(fn + tp), marker='o', label='Bayesian mask ' + name)

    if emp_mask is not None:
        validation_truth = validation_truth.astype(int)
        emp_mask = emp_mask.astype(int)
        tn, fp, fn, tp = (metrics.confusion_matrix(
            validation_truth[:, 0], emp_mask[:, 0], labels=(0, 1))).ravel()
        # print(tn, fp, fn, tp)
        plt.scatter(float(fp) / float(tn + fp), float(tp) /
                    float(fn + tp), marker='*', label='Empirical mask ' + name)

    # Set 2

    if validation_predictions2 is not None:
        if validation_truth2 is not None:
            false_positive_rate2, true_positive_rate2, _ = metrics.roc_curve(
                validation_truth2[:, 0], validation_predictions2[:, 0], pos_label=1)
            plt.plot(false_positive_rate2, true_positive_rate2,
                     label='Model ' + name2)

    if bayes_mask2 is not None:
        validation_truth2 = validation_truth2.astype(int)
        bayes_mask2 = bayes_mask2.astype(int)
        tn, fp, fn, tp = (metrics.confusion_matrix(
            validation_truth2[:, 0], bayes_mask2[:, 0], labels=(0, 1))).ravel()
        # print(tn, fp, fn, tp)
        plt.scatter(float(fp) / float(tn + fp), float(tp) /
                    float(fn + tp), marker='o', label='Bayesian mask ' + name2)

    if emp_mask2 is not None:
        validation_truth2 = validation_truth2.astype(int)
        emp_mask2 = emp_mask2.astype(int)
        tn, fp, fn, tp = (metrics.confusion_matrix(
            validation_truth2[:, 0], emp_mask2[:, 0], labels=(0, 1))).ravel()
        # print(tn, fp, fn, tp)
        plt.scatter(float(fp) / float(tn + fp), float(tp) /
                    float(fn + tp), marker='*', label='Empirical mask ' + name2)

    plt.legend()


def AUC(model, validation_data, validation_truth):
    """Returs area under Receiver Operating Characteristic (ROC) curve"""

    predictions = model.predict(validation_data)

    auc = metrics.roc_auc_score(validation_truth[:, 1], predictions[:, 1])
    return auc


def precision_vs_recall(model, validation_data, validation_truth):
    """Plots precision vs recall curve"""

    predictions = np.nan_to_num(model.predict(validation_data))

    precision, recall, _ = metrics.precision_recall_curve(
        validation_truth[:, 0], predictions[:, 0], pos_label=1)

    plt.figure('Precision vs recall curve')
    plt.title('Precision vs recall curve')
    plt.plot(precision, recall)
    plt.xlabel('Precision')
    plt.ylabel('Recall')


def confusion_matrix(model, validation_data, validation_truth):
    """ Returns a confusion matrix"""

    predictions = model.predict_label(validation_data)
    m = metrics.confusion_matrix(
        validation_truth[:, 0], predictions, labels=(0, 1))
    return m
