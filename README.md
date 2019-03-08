# Masters Project - Cloud Identification in Satellite Images using Artificial Intelligence

A Python library for producing cloud masks of SLSTR scenes using a neural network.

Also includes functions to produce a collocated training data set using either the CALIOP or CATS missions to provide cloud feature labels. In addition, functionality is provided to evaluate the model's performance in different situations using various metrics.

## Overview

![M1](http://www.hep.ph.ic.ac.uk/~trz15/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3/Im1.png?) ![M2](http://www.hep.ph.ic.ac.uk/~trz15/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3/Im2.png?) ![M3](http://www.hep.ph.ic.ac.uk/~trz15/S3A_SL_1_RBT____20180531T222736_20180531T223036_20180602T040456_0179_032_001_1800_LN2_O_NT_003.SEN3/Im3.png?)

This library produces cloud masks which are an improvement over the current standard ESA masks.

## Installation

After cloning this repository and navigating to its install directory, the dependencies can be installed with anaconda:

```conda env create -f environment.yml```

This will create a new anaconda environment 'cloud-id'. After activating the environment, a toggleable matplotlib figure can be produced for an SLSTR S3A_SL_1_RBT folder with:

```python MaskToggle.py PATH```

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.