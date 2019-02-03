import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import DataLoader as DL
import DataPreparation as dp
import Visualisation as Vis
from ffn2 import FFN
from tqdm import tqdm


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
    df = df_model_agreement(path, MaxDist, MaxTime, model, model_network)

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
    df = df_model_agreement(path, MaxDist, MaxTime, model, model_network)

    Vis.plot_poles(df['latitude_an'].values,
                   df['longitude_an'].values, df['Agree'].values)


def df_model_agreement(path, MaxDist, MaxTime, model, model_network):
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


def get_bad_classifications(df):
    """Given a processed dataframe which has model predictions, produce dataframe with poorly classified pixels"""
    bad = df[(df['Agree'] == False) | (
        (df['Label_Confidence'] < 0.7) & (df['Label_Confidence'] > 0.3))]
    return(bad)


def get_contextual_dataframe(df, contextlength=25):
    """Given a dataframe of poorly classified pixels, produce dataframe with neighbouring pixels"""
    # List of all unique SLSTR files in the dataframe
    Sfiles = list(set(df['Sfilename']))

    out = pd.DataFrame()

    for Sfile in tqdm(Sfiles):

        # Load the rows of the dataframe for a SLSTR file
        Sdf = df[df['Sfilename'] == Sfile]

        # Get the indices of the pixels
        Indices = Sdf[['RowIndex', 'ColIndex']].values

        # Get the path to the SLSTR file on the local machine
        Spath = DL.get_SLSTR_path(Sfile)

        # If the file is not on the local machine
        if os.path.exists(Spath) is False:

            # Download the file
            pass

        # Load the SLSTR file
        scn = DL.scene_loader(Spath)

        coords = []

        for i in range(len(Indices)):
            x0, y0 = Indices[i]
            coords += get_coords(x0, y0, contextlength)

        coords = list(set(coords))

        newdf = make_Context_df(coords, Sfile, Spath)

        out = out.append(newdf, ignore_index=True, sort=True)

    return(out)


def get_coords(x0, y0, contextlength):
    East_xs = np.linspace(x0 + 1, x0 + contextlength,
                          contextlength).astype(int)
    West_xs = np.linspace(x0 - contextlength, x0 - 1,
                          contextlength).astype(int)

    North_ys = np.linspace(y0 + 1, y0 + contextlength,
                           contextlength).astype(int)
    South_ys = np.linspace(y0 - contextlength, y0 - 1,
                           contextlength).astype(int)

    # Restrict to indices within 2400, 3000 frame
    East_xs = East_xs[East_xs < 2400]
    West_xs = West_xs[West_xs > -1]
    North_ys = North_ys[North_ys < 3000]
    South_ys = South_ys[South_ys > -1]

    N_list = list(zip([x0] * len(North_ys), North_ys))
    E_list = list(zip(East_xs, [y0] * len(East_xs)))
    S_list = list(zip([x0] * len(South_ys), South_ys))
    W_list = list(zip(West_xs, [y0] * len(West_xs)))

    # Need to reverse South and West so it from x0, y0 outwards
    NE_list = list(zip(East_xs, North_ys))
    SE_list = list(zip(East_xs, South_ys[::-1]))
    NW_list = list(zip(West_xs[::-1], North_ys))
    SW_list = list(zip(West_xs[::-1], South_ys[::-1]))

    return(N_list + E_list + S_list + W_list + NE_list + SE_list + NW_list + SW_list)


def make_Context_df(coords, Sfile, Spath):
    if coords == None:
        return(pd.DataFrame())

    rows = [i[0] for i in coords]
    cols = [i[1] for i in coords]
    num_values = len(coords)

    # Initialise output dataframe
    df = pd.DataFrame()
    # Load SLSTR data and desired attributes
    scn = DL.scene_loader(Spath)
    SLSTR_attributes = ['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an',
                        'S6_an', 'S7_in', 'S8_in', 'S9_in', 'bayes_an',
                        'bayes_bn', 'bayes_cn', 'bayes_in', 'cloud_an',
                        'cloud_bn', 'cloud_cn', 'cloud_in',
                        'satellite_zenith_angle', 'solar_zenith_angle',
                        'latitude_an', 'longitude_an', 'confidence_an']
    scn.load(SLSTR_attributes)

    def Smake_series(Sattribute):
        """
        Make a labelled pandas series for a given SLSTR attribute for all
        matched pixels in an SLSTR file
        """
        # Prepare second index system for data on 1km instead of 0.5km grid
        hrows = [int(i/2) for i in rows]
        hcols = [int(i/2) for i in cols]
        if Sattribute in ['S7_in', 'S8_in', 'S9_in', 'bayes_in', 'cloud_in', 'satellite_zenith_angle', 'solar_zenith_angle']:
            data = scn[Sattribute].values[hrows, hcols]
        else:
            data = scn[Sattribute].values[rows, cols]
        return(pd.Series(data, name=Sattribute))

    for attribute in SLSTR_attributes:
        df = df.append(Smake_series(attribute))

    Sfilenameser = pd.Series([Sfile] * num_values, name='Sfilename')
    rowser = pd.Series(rows, name='RowIndex')
    colser = pd.Series(cols, name='ColIndex')


    df = df.append(Sfilenameser)
    df = df.append(rowser)
    df = df.append(colser)

    df = df.transpose()

    df.columns = SLSTR_attributes + ['Sfilename', 'RowIndex', 'ColIndex']
    return(df)
