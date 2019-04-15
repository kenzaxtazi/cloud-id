##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

from FFNtest import FFN
from tqdm import tqdm
import DataPreparation as dp
import tensorflow as tf
import ModelEvaluation as me
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


# This is a script to get some visulisation of unbias hyperparameter testing. 

df = dp.PixelLoader('./SatelliteData/Pixels3')

tdata, vdata, ttruth, vtruth = df.dp.get_ffn_training_data(21)


epochs = [1]
neurons = [32, 16, 64, 128, 254, 512, 1024]
LRs = [1e-3]  # , 1e-1, 1e-2, 1e-4]
hidden_layers = [4, 8, 16, 32, 64, 128, 254]
batch_size = [64]  # 16, 32, 128]
dropout = [0.8]  # , 0.2, 0.4, 0.6]

hyp_df = pd.DataFrame(columns=['epochs', 'neurons', 'hidden_layers', 'batch_size', 'LR', 'dropout', 'val_acc'])

for e in epochs:
    for bs in batch_size:
        for lr in tqdm(LRs):
            for do in dropout:
                for hl in tqdm(hidden_layers):
                    for n in neurons:
                        model = FFN(str(e) + '_' + str(bs) + '_' + str(lr) + '_' +
                                    str(do) + '_' + str(hl) + '_' + str(n), 'TestNetwork', 21, LR=lr,
                                    neuron_num=n, hidden_layers=hl, batch_size=bs,
                                    epochs=e, dout=do)
                        model.Train(tdata, ttruth, vdata, vtruth)
                        val_acc = me.get_accuracy(model.model, vdata, vtruth)
                        hyp_df = hyp_df.append({'epochs': e, 'neurons': n,
                                                'hidden_layers': hl, 'batch_size': bs,
                                                'LR': lr, 'dropout': do, 'val_acc': val_acc}, 
                                               ignore_index=True)
                        model.Save()
                        tf.reset_default_graph()

df.to_pickle('hyp_test')


# Fig 2

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = neurons
Y = hidden_layers
X, Y = np.meshgrid(X, Y)
fined_hyp_df = hyp_df[hyp_df['neurons'] == X]
refined_hyp_df = fined_hyp_df[fined_hyp_df['hidden_layers'] == Y]
Z = refined_hyp_df['val_acc'].values 

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf)
plt.show()
