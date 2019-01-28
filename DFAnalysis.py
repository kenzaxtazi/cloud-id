import matplotlib.pyplot as plt
import pandas as pd

import DataPreparation as dp
import Visualisation as Vis
from ffn2 import FFN


def make_confidence_hist(path, model='Net3_S_FFN', model_network='Network2', MaxDist=500, MaxTime=1200):
    """
    Makes a histogram of the model confidence for correctly and incorrectly classified pixels in a given directory or .pkl file.

    Parameters
    ----------
    path: str
        Path to a .pkl pandas dataframe file or a directory containing them

    model: str
        Name of a FFN model saved in the Models/ subdirectory
        Default is 'Net3_S_FFN'

    model_network: str
        Name of the network configuration to use with the given model
        Default is 'Network2'

    MaxDist: int or float
        Maximum accepted distance between collocated pixels in dataframe to consider

    MaxTime: int or float
        Maximum accepted time difference between collocated pixels in dataframe to consider
    """
    df = _df_model_agreement(path, MaxDist, MaxTime, model, model_network)

    bad = df[df['Agree'] == False]

    bconfidence = bad['Label_Confidence'].values
    tconfidence = df['Label_Confidence'].values

    plt.hist(tconfidence, 250, label='Certainty of model for all predictions')
    plt.hist(bconfidence, 250, label='Certainty of model for incorrect predictions')
    plt.legend()
    plt.title('Histogram of model prediction certainty for collocated data')
    plt.xlim([0, 1])
    plt.show()


def plot_pixels(path, model='Net3_S_FFN', model_network='Network2', MaxDist=500, MaxTime=1200):
    """
    Plots the correctly and incorrectly classified pixels in a given directory or .pkl file.

    Parameters
    ----------
    path: str
        Path to a .pkl pandas dataframe file or a directory containing them

    model: str
        Name of a FFN model saved in the Models/ subdirectory
        Default is 'Net3_S_FFN'

    model_network: str
        Name of the network configuration to use with the given model
        Default is 'Network2'

    MaxDist: int or float
        Maximum accepted distance between collocated pixels in dataframe to consider

    MaxTime: int or float
        Maximum accepted time difference between collocated pixels in dataframe to consider
    """
    df = _df_model_agreement(path, MaxDist, MaxTime, model, model_network)

    Vis.plot_poles(df['latitude_an'].values,
                   df['longitude_an'].values, df['Agree'].values)


def _df_model_agreement(path, MaxDist, MaxTime, model, model_network):
    # Add useful columns to a dataframe generated from data at path
    if path.endswith('.pkl'):
        df = pd.read_pickle(path)
    else:
        df = dp.PixelLoader(path)
    df = df[df['Distance'] < MaxDist]
    df = df[abs(df['TimeDiff']) < MaxTime]

    inputs = dp.inputs_from_df(df, 24)
    model = FFN(model, model_network)
    model.Load()

    output_labels = model.model.predict_label(inputs)
    output_con = model.model.predict(inputs)

    df['Labels'] = pd.Series(output_labels[:, 0], index=df.index)
    df['Label_Confidence'] = pd.Series(output_con[:, 0], index=df.index)

    dp.make_CTruth_col(df)

    df['Agree'] = df['CTruth'] != df['Labels']
    return df
