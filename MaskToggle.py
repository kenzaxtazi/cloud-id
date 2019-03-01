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


class MaskToggler():
    def __init__(self, Sfilename, model='Net1_FFN_v7', verbose=False):
        self.index = 0
        self.settingfuncs = [self.setting1, self.setting2,
                             self.setting3, self.setting4, self.setting5]
        if isinstance(model, str):
            self.modelname = model

            self.model = FFN(model)
            self.model.Load(verbose=verbose)

        elif isinstance(model, FFN):
            pass

        mask1, pmask = self.model.apply_mask(Sfilename)

        rgb, self.TitleStr = Vis.FalseColour(Sfilename, False)

        self.im1 = plt.imshow(rgb)
        plt.title('False colour image\n' + self.TitleStr)

        self.im2 = plt.imshow(mask1, cmap='Blues')
        self.im2.set_visible(False)

        bmask = DL.extract_mask(Sfilename, 'bayes_in', 2)
        self.im3 = plt.imshow(bmask, cmap='Reds')
        self.im3.set_visible(False)

        mask1 = mask1.astype('bool')
        rgb[~mask1, 0] = 254 / 255
        rgb[~mask1, 1] = 253 / 255
        rgb[~mask1, 2] = 185 / 255
        self.im4 = plt.imshow(rgb)
        self.im4.set_visible(False)

        self.im5 = plt.imshow(1 - pmask, cmap='Oranges')
        self.im5.set_visible(False)

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
        self.im1.set_visible(False)
        self.im2.set_visible(False)
        self.im3.set_visible(False)
        self.im4.set_visible(False)
        self.im5.set_visible(False)

    def cycleforward(self):
        self.index = (self.index + 1) % 5
        self.settingfuncs[self.index]()

    def cyclebackward(self):
        self.index = (self.index - 1) % 5
        self.settingfuncs[self.index]()

    def setting1(self):
        plt.title('False colour image\n' + self.TitleStr)
        self.im1.set_visible(True)
        self.index = 0
        plt.draw()

    def setting2(self):
        plt.title(self.modelname + ' mask\n' + self.TitleStr)
        self.im2.set_visible(True)
        self.index = 1
        plt.draw()

    def setting3(self):
        plt.title('Bayesian mask\n' + self.TitleStr)
        self.im3.set_visible(True)
        self.index = 2
        plt.draw()

    def setting4(self):
        plt.title(self.modelname +
                  ' masked false colour image\n' + self.TitleStr)
        self.im4.set_visible(True)
        self.index = 3
        plt.draw()

    def setting5(self):
        plt.title(self.modelname + ' probability mask\n' + self.TitleStr)
        self.im5.set_visible(True)
        self.index = 4
        plt.draw()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        Sfile = r"./SatelliteData/SLSTR/2018/05/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3"
    else:
        Sfile = sys.argv[1]

    toggler = MaskToggler(Sfile)

    plt.connect('key_press_event', toggler.toggle_images)

    plt.show()
