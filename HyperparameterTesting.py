##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

from FFN import FFN
from tqdm import tqdm
import DataPreparation as dp
import tensorflow as tf


# Correlation test

df = dp.PixelLoader('./SatelliteData/Pixels3')

tdata, vdata, ttruth, vtruth = df.dp.get_ffn_training_data(21)

epochs = [10, 50, 100, 150]
neurons = [16, 32, 64, 128]
LRs = [1e-1, 1e-2, 1e-3, 1e-4]
hidden_layers = [4, 8, 16, 32]
batch_size = [16, 32, 64, 128]
dropout = [0.2, 0.4, 0.6, 0.8]


for e in epochs:
    for hl in hidden_layers:
        for n in neurons:
            for bs in batch_size:
                for lr in tqdm(LRs):
                    for do in dropout:
                        model = FFN(str(e) + '_' + str(n) + '_' + str(hl) + '_' +
                                    str(bs) + '_' + str(lr) + '_' + str(do), 'TestNetwork', 21, LR=lr,
                                    neuron_num=n, hidden_layers=hl, batch_size=bs,
                                    epochs=e, dout=do)
                        model.Train(tdata, ttruth, vdata, vtruth)
                        model.Save()
                        tf.reset_default_graph()

# Further investigation
