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


This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.