from ModelApplication import apply_mask
import DataLoader as DL
import Visualisation as Vis
from FFN import FFN

model = FFN('Net2_S_FFN', 'Network2')
model.Load()


# Pacific Ocean
Sfile = r"D:\SatelliteData\SLSTR\Dataset1\S3A_SL_1_RBT____20180528T190110_20180528T190410_20180529T235914_0179_031_341_2880_LN2_O_NT_003.SEN3"
# Sfile = r"D:\SatelliteData\SLSTR\Dataset1\S3A_SL_1_RBT____20180817T190106_20180817T190406_20180819T002616_0179_034_341_2880_LN2_O_NT_003.SEN3"

# NE Australia
# Sfile = r"D:\SatelliteData\SLSTR\Dataset1\S3A_SL_1_RBT____20180822T000319_20180822T000619_20180823T043035_0179_035_016_3060_LN2_O_NT_003.SEN3"
# Sfile = r"D:\SatelliteData\SLSTR\Dataset1\S3A_SL_1_RBT____20180822T000619_20180822T000919_20180822T015223_0179_035_016_3240_SVL_O_NR_003.SEN3"

# Mainland Australia
# Sfile = r"D:\SatelliteData\SLSTR\Dataset1\S3A_SL_1_RBT____20180822T000919_20180822T001219_20180822T015515_0179_035_016_3420_SVL_O_NR_003.SEN3"

mask1 = apply_mask(model.model, Sfile)[0]

# bmask = DL.extract_mask(Sfile, 'cloud_an', 64)
bmask = DL.extract_mask(Sfile, 'bayes_in', 2)

Vis.MaskComparison(Sfile, mask1, bmask, True, 1000)
