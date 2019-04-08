##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import numpy as np
import ModelEvaluation as me
from FFN import FFN
from tqdm import tqdm
import DataPreparation as dp


# Correlation test

df = dp.PixelLoader('./SatelliteData/SLSTR/Pixels3')

tdata, vdata, ttruth, vtruth = df.dp.get_ffn_training_data(21)

LRs = [1e-1, 1e-2, 1e-3, 1e-4]
neurons = [16, 32, 64, 128]
hidden_layers = [2, 4, 6, 8, 10]
batch_size = [16, 32, 64, 128]
epochs = [10, 50, 100, 150]
dropout = [0.2, 0.4, 0.6, 0.8]

val_accuracy = []


for lr in tqdm(LRs):
    for n in neurons:
        for hl in hidden_layers:
            for bs in batch_size:
                for e in epochs:
                    for do in dropout:
                        model = FFN(str(lr) + '_' + str(n) + '_' + str(hl) + '_' + 
                                    str(bs) + '_' + str(do),'TestNetwork', 21, LR=lr,
                                    neuron_num=n, hidden_layers=hl, batch_size=bs, 
                                    epochs=e, dout=do)
                        model.Train(tdata, ttruth, vdata, vtruth)
                        acc = me.get_accuracy(model, vdata, vtruth, para_num=21)
                        val_accuracy.append(acc)
                        model.Save()

# Further investigation
