
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import sys

import matplotlib.pyplot as plt

import DataLoader as DL
import Visualisation as Vis
from FFN import FFN

model = FFN('Net1_FFN_v7')
model.Load()
modelname = model.name

if __name__ == '__main__':
    if len(sys.argv) == 1:
        Sfilename = r"./SatelliteData/SLSTR/2018/05/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3"
    else:
        Sfilename = sys.argv[1]

    mask1, pmask = model.apply_mask(Sfilename)

    rgb, TitleStr = Vis.FalseColour(Sfilename, False)

    plt.figure()
    plt.imshow(rgb)
    plt.title('False colour image\n' + TitleStr)

    plt.figure()
    plt.imshow(mask1, cmap='Blues')
    plt.title(modelname + ' mask\n' + TitleStr)

    plt.figure()
    bmask = DL.extract_mask(Sfilename, 'bayes_in', 2)
    im3 = plt.imshow(bmask, cmap='Reds')
    plt.title('Bayesian mask\n' + TitleStr)

    plt.figure()
    mask1 = mask1.astype('bool')
    rgb[~mask1, :] = 254 / 255, 253 / 255, 185 / 255
    im4 = plt.imshow(rgb)
    plt.title(modelname + ' masked false colour image\n' + TitleStr)

    plt.figure()
    im5 = plt.imshow(1 - pmask, cmap='Oranges')
    plt.title(modelname + ' model output\n' + TitleStr)
    plt.colorbar(im5)

    plt.figure()
    maskdiff = bmask - mask1
    im6 = plt.imshow(maskdiff, cmap='bwr')
    plt.title(modelname + ' mask - Bayesian mask\n' + TitleStr)
    plt.colorbar(im6)

    plt.show()
