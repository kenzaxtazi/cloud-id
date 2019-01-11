# Masters Project

### Cloud Identification in Satellite Images using Artificial Intelligence

Model version 0.1.0 is available to use.

Dependencies are listed in requirements.txt. Satpy may require installation of an old version with modifications.

//TODO Update Satpy version dependency and check compatability

Usage instructions:

Cloud mask algorithm can be run from MaskSLSTR.py. Assign a valid path to an SLSTR SL_1_RBT___ product folder to the Sfile variable.

File can then be run as script, e.g.

python MaskSLSTR.py

Valid SLSTR files are available from https://scihub.copernicus.eu/s3/#/home
