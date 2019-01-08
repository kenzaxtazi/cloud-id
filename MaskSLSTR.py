from ModelApplication import apply_mask
from ModelLoader import model_loader

Sfile = r"D:\SatelliteData\SLSTR\PacificTest\S3A_SL_1_RBT____20180209T190102_20180209T190402_20180210T234449_0179_027_341_2880_LN2_O_NT_002.SEN3"

model = model_loader()
apply_mask(model, Sfile)
