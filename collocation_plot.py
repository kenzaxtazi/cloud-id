#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:43:57 2018

@author: kenzatazi
"""

from satpy import Scene
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import DataLoader as DL
import vfm_feature_flags2 as vfm
import CalipsoReader2 as CR
import DataLoader as DL
from Collocation2 import collocate


def plotmatchingpixels(Sfilename, Cfilename, coords):
    scn = DL.scene_loader(Sfilename)
    scn.load(['S1_an'])
    S1 = np.nan_to_num(scn['S1_an'].values)
    with CR.SDopener(Cfilename) as file:
        data = CR.load_data(file, 'Feature_Classification_Flags')
    values = []

    for i in coords[:, 2]:
        # see vfm function for value meaning
        values.append(float(vfm.vfm_feature_flags((data[i, 0]))))
    m = cm.ScalarMappable(cmap=cm.get_cmap('jet'))
    m.set_array(values)
    plt.figure()
    plt.imshow(S1, 'gray')
    plt.scatter(coords[:, 1], coords[:, 0], c=values, cmap='jet')
    plt.colorbar(m)
    plt.show()


if __name__ == '__main__':
    Cfilename = "D:/SatelliteData/Calipso1km/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T00-04-48ZD.hdf"
    Sfilename = "D:/SatelliteData/SLSTR/S3A_SL_1_RBT____20180401T012743_20180401T013043_20180402T055007_0179_029_288_1620_LN2_O_NT_002.SEN3"
    coords = np.array(collocate(Sfilename, Cfilename))

    plotmatchingpixels(Sfilename, Cfilename, coords)
