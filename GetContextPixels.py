
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import DFAnalysis2 as DFA


df = DFA.df_model_agreement('/home/hep/trz15/Matched_Pixels2/Calipso/P4',
                            MaxDist=1000000, MaxTime=1000000, model='Net1_FFN', model_network='Network1')

bad = DFA.get_bad_classifications(df)

context_df = DFA.get_contextual_dataframe(bad)

Ints = ['ColIndex', 'RowIndex', 'bayes_an', 'bayes_bn', 'bayes_cn',
        'bayes_in', 'cloud_an', 'cloud_bn', 'cloud_cn', 'cloud_in', 'confidence_an']
Flos = ['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in',
        'satellite_zenith_angle', 'solar_zenith_angle', 'latitude_an', 'longitude_an']

context_df[Ints] = context_df[Ints].astype('int')
context_df[Flos] = context_df[Flos].astype('float')


context_df.to_pickle('ContextualPixels.pkl')
