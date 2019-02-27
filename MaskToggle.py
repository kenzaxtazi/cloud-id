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

if len(sys.argv) == 1:
    Sfile = r"./SatelliteData/SLSTR/2018/05/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3"
else:
    Sfile = sys.argv[1]

modelname = 'Net1_FFN_v7'
model = FFN(modelname)
model.Load()

mask1, pmask = model.apply_mask(Sfile)

rgb, TitleStr = Vis.FalseColour(Sfile, False)

im1 = plt.imshow(rgb)
plt.title('False colour image\n' + TitleStr)

im2 = plt.imshow(mask1, cmap='Blues')
im2.set_visible(False)

bmask = DL.extract_mask(Sfile, 'bayes_in', 2)
im3 = plt.imshow(bmask, cmap='Reds')
im3.set_visible(False)

mask1 = mask1.astype('bool')
rgb[~mask1, 0] = 254 / 255
rgb[~mask1, 1] = 253 / 255
rgb[~mask1, 2] = 185 / 255
im4 = plt.imshow(rgb)
im4.set_visible(False)

im5 = plt.imshow(1 - pmask, cmap='Oranges')
im5.set_visible(False)


class Toggler():
    def __init__(self):
        self.index = 0
        self.settingfuncs = [self.setting1, self.setting2,
                             self.setting3, self.setting4, self.setting5]

    def toggle_images(self, event):
        """Toggle between different images to display"""
        if event.key == '1':
            self._clearframe()
            self.setting1()
        elif event.key == '2':
            self._clearframe()
            self.setting2()
        elif event.key == '3':
            self._clearframe()
            self.setting3()
        elif event.key == '4':
            self._clearframe()
            self.setting4()
        elif event.key == '5':
            self._clearframe()
            self.setting5()
        elif event.key == 'm':
            self._clearframe()
            self.cycleforward()
        elif event.key == 'n':
            self._clearframe()
            self.cyclebackward()
        else:
            return

    def _clearframe(self):
        im1.set_visible(False)
        im2.set_visible(False)
        im3.set_visible(False)
        im4.set_visible(False)
        im5.set_visible(False)

    def cycleforward(self):
        self.index = (self.index + 1) % 5
        self.settingfuncs[self.index]()

    def cyclebackward(self):
        self.index = (self.index - 1) % 5
        self.settingfuncs[self.index]()

    def setting1(self):
        plt.title('False colour image\n' + TitleStr)
        im1.set_visible(True)
        self.index = 0
        plt.draw()

    def setting2(self):
        plt.title(modelname + ' mask\n' + TitleStr)
        im2.set_visible(True)
        self.index = 1
        plt.draw()

    def setting3(self):
        plt.title('Bayesian mask\n' + TitleStr)
        im3.set_visible(True)
        self.index = 2
        plt.draw()

    def setting4(self):
        plt.title(modelname + ' masked false colour image\n' + TitleStr)
        im4.set_visible(True)
        self.index = 3
        plt.draw()

    def setting5(self):
        plt.title(modelname + ' probability mask\n' + TitleStr)
        im5.set_visible(True)
        self.index = 4
        plt.draw()


toggler = Toggler()

plt.connect('key_press_event', toggler.toggle_images)

plt.show()
