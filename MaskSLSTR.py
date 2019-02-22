
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import DataLoader as DL
import Visualisation as Vis
from FFN import FFN

model = FFN('Net1_FFN_v4')
model.Load()


# Pacific Ocean
# Sfile = "./SatelliteData/SLSTR/Datas/t1\S3A_SL_1_RBT____20180528T190110_20180528T190410_20180529T235914_0179_031_341_2880_LN2_O_NT_003.SEN3"
# Sfile = "./SatelliteData/SLSTR/Datas/t1\S3A_SL_1_RBT____20180817T190106_20180817T190406_20180819T002616_0179_034_341_2880_LN2_O_NT_003.SEN3"

# NE Australia
# Sfile = "./SatelliteData/SLSTR/Datas/t1\S3A_SL_1_RBT____20180822T000319_20180822T000619_20180823T043035_0179_035_016_3060_LN2_O_NT_003.SEN3"
# Sfile = "./SatelliteData/SLSTR/Datas/t1\S3A_SL_1_RBT____20180822T000619_20180822T000919_20180822T015223_0179_035_016_3240_SVL_O_NR_003.SEN3"

# Mainland Australia
# Sfile = "./SatelliteData/SLSTR/Datas/t1\S3A_SL_1_RBT____20180822T000919_20180822T001219_20180822T015515_0179_035_016_3420_SVL_O_NR_003.SEN3"

Sfiles = ["./SatelliteData/SLSTR/Test4/S3A_SL_1_RBT____20180212T103733_20180212T104033_20180213T142548_0180_027_379_1620_LN2_O_NT_002.SEN3",
          "./SatelliteData/SLSTR/Test4/S3A_SL_1_RBT____20180503T041955_20180503T042255_20180504T103326_0180_030_361_1620_LN2_O_NT_003.SEN3",
          "./SatelliteData/SLSTR/Test4/S3A_SL_1_RBT____20180908T072307_20180908T072607_20180909T124616_0179_035_263_1620_LN2_O_NT_003.SEN3",
          "./SatelliteData/SLSTR/Test4/S3A_SL_1_RBT____20180911T010137_20180911T010437_20180912T055246_0179_035_302_1620_LN2_O_NT_003.SEN3",
          "./SatelliteData/SLSTR/Test4/S3A_SL_1_RBT____20181004T225815_20181004T230115_20181006T034958_0179_036_258_1620_LN2_O_NT_003.SEN3"]


for Sfile in Sfiles:

    mask1 = model.apply_mask(Sfile)[0]

    # bmask = DL.extract_mask(Sfile, 'cloud_an', 64)
    bmask = DL.extract_mask(Sfile, 'bayes_in', 2)

    Vis.MaskComparison(Sfile, mask1, bmask, True, 1000)
