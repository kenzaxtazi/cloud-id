import matplotlib.pyplot as plt

import DataLoader as DL
import Visualisation as Vis
from FFN import FFN

Sfile = r"./SatelliteData/SLSTR/Poster/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3"

modelname = 'Net1_FFN_v7'
model = FFN(modelname)
model.Load()

mask1 = model.apply_mask(Sfile)[0]

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


# TODO Refactor
class Toggler():
    def __init__(self):
        self.setting = 1

    def toggle_images(self, event):
        """Toggle between different images to display"""
        if event.key == '1':
            self.setting1()
        elif event.key == '2':
            self.setting2()
        elif event.key == '3':
            self.setting3()
        elif event.key == '4':
            self.setting4()
        else:
            return

    def setting1(self):
        plt.title('False colour image\n' + TitleStr)
        im1.set_visible(True)
        im2.set_visible(False)
        im3.set_visible(False)
        im4.set_visible(False)
        self.setting = 1
        plt.draw()

    def setting2(self):
        plt.title(modelname + ' mask\n' + TitleStr)
        im1.set_visible(False)
        im2.set_visible(True)
        im3.set_visible(False)
        im4.set_visible(False)
        self.setting = 2
        plt.draw()

    def setting3(self):
        plt.title('Bayesian mask\n' + TitleStr)
        im1.set_visible(False)
        im2.set_visible(False)
        im3.set_visible(True)
        im4.set_visible(False)
        self.setting = 3
        plt.draw()

    def setting4(self):
        plt.title(modelname + ' masked false colour image\n' + TitleStr)
        im1.set_visible(False)
        im2.set_visible(False)
        im3.set_visible(False)
        im4.set_visible(True)
        self.setting = 4
        plt.draw()


toggler = Toggler()

plt.connect('key_press_event', toggler.toggle_images)

plt.show()
