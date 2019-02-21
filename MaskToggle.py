from ModelApplication import apply_mask
import DataLoader as DL
import Visualisation as Vis
from FFN import FFN

import matplotlib.pyplot as plt
import numpy as np

model = FFN('Net1_FFN_v7')
model.Load()

Sfile = r"C:\SLSTR\S3A_SL_1_RBT____20180703T235729_20180704T000029_20180704T012919_0179_033_087_4500_SVL_O_NR_003.SEN3"

mask1 = apply_mask(model.model, Sfile, input_type=22)[0]
mask1 = 1 - mask1

# two images x1 is initially visible, x2 is not

rgb = Vis.FalseColour(Sfile, False)[0]

im1 = plt.imshow(rgb)
im2 = plt.imshow(mask1, cmap='Blues')
im2.set_visible(False)


def toggle_images(event):
    'toggle the visible state of the two images'
    if event.key != 't':
        return
    b1 = im1.get_visible()
    b2 = im2.get_visible()
    im1.set_visible(not b1)
    im2.set_visible(not b2)
    plt.draw()

plt.connect('key_press_event', toggle_images)

plt.show()
