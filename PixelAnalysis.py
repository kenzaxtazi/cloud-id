import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def PixelLoader(directory):
    """Load all pixels in a directory of pickle files into a single DataFrame"""

    if directory.endswith('/') is False:
        directory += '/'
    PickleFilenames = os.listdir(directory)
    PicklePaths = [directory + i for i in PickleFilenames]

    out = pd.DataFrame()
    for file in PicklePaths:
        out = out.append(pd.read_pickle(file), sort=True, ignore_index=True)

    print("%s pixels loaded" % (len(out)))
    return(out)


def truth_from_bitmask(row):
    val = row['Feature_Classification_Flags']
    val = np.nan_to_num(val)
    val = int(val) & 7
    if val == 2:
        return(True)
    else:
        return(False)


def make_attrib_hist(df, column='Latitude'):
    out = df[column]
    frq, edges = np.histogram(out, 100)
    plt.title(column + ' histogram')
    plt.bar(edges[:-1], frq, width=np.diff(edges), ec='k', align='edge')
    plt.show()


def make_CTruth_col(df):
    tqdm.pandas()

    def flagtruths(row):
        val = row['Feature_Classification_Flags']
        val = np.nan_to_num(val)
        val = int(val) & 7
        if val == 2:
            return(True)
        else:
            return(False)
    df['CTruth'] = df.progress_apply(lambda row: flagtruths(row), axis=1)
    return(df)


def make_STruth_col(df, cloudmask='cloud_an', bit=1):
    tqdm.pandas()

    def flagtruths(row):
        val = row[cloudmask]
        val = np.nan_to_num(val)
        val = int(val) & bit
        if val == bit:
            return(True)
        else:
            return(False)
    df['STruth'] = df.progress_apply(lambda row: flagtruths(row), axis=1)
    return(df)


def TruthMatches(df):
    q = df['CTruth'] == df['STruth']
    out = np.mean(q)
    return(out)


# PickleDirectory = "D:/Users/tomzh/Desktop/Pixels/"

# df = PixelLoader(PickleDirectory)
# df = df[abs(df['TimeDiff']) < 200]

# df = make_CTruth_col(df)


# for bit in range(1, 17):
#     df = make_STruth_col(df, 'cloud_an', bit)
#     print("Bit: %s" % (bit))
#     print("Fraction of matches: %s" % (TruthMatches(df)))
