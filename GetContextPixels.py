
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import DataPreparation as dp
import DataAnalyser  # Needed for 'da' dataframe accessor

df = dp.PixelLoader('./SatelliteData/SLSTR/Pixels4/')

context_df = df.da.get_contextual_dataframe(
    contextlength=50, download_missing=False)

context_df.to_pickle('Run1.pkl')
