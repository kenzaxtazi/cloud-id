from ModelApplication import apply_mask
from ModelLoader import model_loader
import DataLoader as DL
import Visualisation as Vis

Sfile = r"D:\SatelliteData\SLSTR\S3A_SL_1_RBT____20180822T000919_20180822T001219_20180822T015515_0179_035_016_3420_SVL_O_NR_003.SEN3"

model = model_loader()

mask1 = apply_mask(model, Sfile)
bmask = DL.extract_mask(Sfile, 'bayes_in', 2)

Vis.MaskComparison(Sfile, mask1, bmask)