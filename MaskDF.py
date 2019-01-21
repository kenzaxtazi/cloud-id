import PixelAnalysis as PA
from ModelLoader import model_loader

df = PA.PixelLoader(r'D:\SatelliteData\SLSTR\Pixels2')
inputs = PA.inputs_from_df(df)
model = model_loader()

ModelOutputs = model.predict_label(inputs)
ModelConfidence = model.predict(inputs)
