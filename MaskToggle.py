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
        self.settingfuncs = [self.setting1, self.setting2, self.setting3,
                             self.setting4, self.setting5, self.setting6,
                             self.setting7, self.setting8]
        if isinstance(model, str):
            self.modelname = model

            self.model = FFN(model)
            self.model.Load(verbose=verbose)

        elif isinstance(model, FFN):
            self.model = model
            self.modelname = self.model.name

        mask1, pmask1 = self.model.apply_mask(Sfilename)

        rgb, self.TitleStr = Vis.FalseColour(Sfilename, False)

        scn = DL.scene_loader(Sfilename)
        scn.load(['bayes_in', 'probability_cloud_single_in'])
        bmask = DL.upscale_repeat(scn['bayes_in'].values).astype('int')
        bmask = 1 - ((bmask & 2) / 2)
        bpmask = DL.upscale_repeat(scn['probability_cloud_single_in'].values)

        self.im1 = plt.imshow(rgb)
        plt.title('False colour image\n' + self.TitleStr)

        self.im2 = plt.imshow(mask1, cmap='Blues')
        self.im2.set_visible(False)

        bmask = DL.extract_mask(Sfilename, 'bayes_in', 2)
        self.im3 = plt.imshow(bmask, cmap='Reds')
        self.im3.set_visible(False)

        mask1 = mask1.astype('bool')
        temp = rgb
        temp[~mask1, :] = 254 / 255, 253 / 255, 185 / 255
        self.im4 = plt.imshow(temp)
        self.im4.set_visible(False)

        rgb[mask1, :] = 74 / 255, 117 / 255, 50 / 255
        self.im5 = plt.imshow(rgb)
        self.im5.set_visible(False)

        self.im6 = plt.imshow(1 - pmask1, cmap='Oranges')
        self.im6.set_visible(False)

        self.im7 = plt.imshow(1 - bpmask, cmap='Reds')
        self.im7.set_visible(False)

        maskdiff = bmask - mask1
        self.im8 = plt.imshow(maskdiff, cmap='bwr')
        self.im8.set_visible(False)

        self.cbset = False
        self.cb = None

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
        elif event.key == '6':
            self._clearframe()
            self.setting6()
        elif event.key == '7':
            self._clearframe()
            self.setting7()
        elif event.key == '8':
            self._clearframe()
            self.setting8()
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
        self.im6.set_visible(False)
        self.im7.set_visible(False)
        self.im8.set_visible(False)
        if self.cbset:
            self.cb.remove()
            self.cbset = False

    def cycleforward(self):
        self.index = (self.index + 1) % 8
        self.settingfuncs[self.index]()

    def cyclebackward(self):
        self.index = (self.index - 1) % 8
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
        plt.title(self.modelname
                  + ' masked false colour image\n' + self.TitleStr)
        self.im4.set_visible(True)
        self.index = 3
        plt.draw()

    def setting5(self):
        plt.title(self.modelname
                  + ' reverse masked false colour image\n' + self.TitleStr)
        self.im5.set_visible(True)
        self.index = 4
        plt.draw()

    def setting6(self):
        plt.title(self.modelname + ' raw model output\n' + self.TitleStr)
        self.im6.set_visible(True)
        self.index = 5
        self.cb = plt.colorbar(self.im6)
        self.cbset = True
        plt.draw()

    def setting7(self):
        plt.title('Bayesian mask raw output\n' + self.TitleStr)
        self.im7.set_visible(True)
        self.index = 6
        self.cb = plt.colorbar(self.im7)
        self.cbset = True
        plt.draw()

    def setting8(self):
        plt.title(self.modelname + ' mask - Bayesian mask\n' + self.TitleStr)
        self.im8.set_visible(True)
        self.index = 7
        self.cb = plt.colorbar(self.im8)
        self.cbset = True
        plt.draw()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        Sfile = r"./SatelliteData/SLSTR/2018/05/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3"
    else:
        Sfile = sys.argv[1]

    toggler = MaskToggler(Sfile)

    plt.connect('key_press_event', toggler.toggle_images)

    plt.show()
