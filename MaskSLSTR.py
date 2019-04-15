
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import DataLoader as DL
import Visualisation as Vis
from FFN import FFN

plt.rcParams.update({'font.size': 22})

def mask_debug(Sfilename, model, verbose):
    if isinstance(model, str):
        modelname = model

        model = FFN(model)
        model.Load(verbose=verbose)

    elif isinstance(model, FFN):
        modelname = model.name

    mask1, pmask = model.apply_mask(Sfilename)

    rgb, TitleStr = Vis.FalseColour(Sfilename, False)

    plt.figure()
    plt.imshow(rgb)
    plt.title('False colour image\n' + TitleStr)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000],
               [0, 250, 500, 750, 1000, 1250, 1500])
    plt.yticks([0, 500, 1000, 1500, 2000], [0, 250, 500, 750, 1000])

    plt.figure()
    plt.imshow(mask1, cmap='Blues')
    plt.title(modelname + ' mask\n' + TitleStr)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000],
               [0, 250, 500, 750, 1000, 1250, 1500])
    plt.yticks([0, 500, 1000, 1500, 2000], [0, 250, 500, 750, 1000])

    plt.figure()
    bmask = DL.extract_mask(Sfilename, 'bayes_in', 2)
    plt.imshow(bmask, cmap='Reds')
    plt.title('Bayesian mask\n' + TitleStr)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000],
               [0, 250, 500, 750, 1000, 1250, 1500])
    plt.yticks([0, 500, 1000, 1500, 2000], [0, 250, 500, 750, 1000])

    plt.figure()
    mask1 = mask1.astype('bool')
    rgb[~mask1, :] = 254 / 255, 253 / 255, 185 / 255
    plt.imshow(rgb)
    plt.title(modelname + ' masked false colour image\n' + TitleStr)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000],
               [0, 250, 500, 750, 1000, 1250, 1500])
    plt.yticks([0, 500, 1000, 1500, 2000], [0, 250, 500, 750, 1000])
    
    plt.figure()
    ax = plt.gca()
    im5 = plt.imshow(pmask, cmap='Oranges_r')
    plt.title(modelname + ' model output\n' + TitleStr)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im5, cax=cax)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000],
               [0, 250, 500, 750, 1000, 1250, 1500])
    plt.yticks([0, 500, 1000, 1500, 2000], [0, 250, 500, 750, 1000])

    plt.figure()
    maskdiff = mask1 - bmask
    ax = plt.gca()
    im6 = plt.imshow(maskdiff, cmap='bwr_r')
    plt.title(modelname + ' mask - Bayesian mask\n' + TitleStr)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im6, cax=cax)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000],
               [0, 250, 500, 750, 1000, 1250, 1500])
    plt.yticks([0, 500, 1000, 1500, 2000], [0, 250, 500, 750, 1000])

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        Sfile = r"./SatelliteData/SLSTR/2018/05/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3"
    else:
        Sfile = sys.argv[1]

    mask_debug(Sfile, 'Net1_FFN_v7', True)
