# Material for meeting on 21/11/18

We are succesfully training our network on CALIOP data. 

### Fixes from last week:

Tom has installed Anaconda onto lx02 to succesfully get pyhdf working.
* Attempted to follow the instructions to install without anaconda
* However, he couldn't find any compiled binaries, and all compile attempts failed. Traceback gave error about recompiling files in a CERN directory with fPIC.

Which Python and which pip show that anaconda is properly installed.

In addition, the time discrepency has been fixed.
* 10 seconds were from leap seconds added since Jan 1993
* 1 hour is because the method used was .fromtimestamp instead of .utcfromtimestamp. The method returns the local date which was BST. 

### Data processing
Tom rewrote the dhusget.sh script in python using requests.
* The bash script always run verbosely, attempts to output to /dev/null failed
* The bash script had strange behaviour, such as always making a failed connection attempt to some address

The [XML](https://scihub.copernicus.eu/s3//search?q=%20instrumentshortname:SLSTR%20AND%20producttype:SL_1_RBT___%20AND%20(%20footprint:%22Intersects(POLYGON((101.1878500000000%2016.7101170000000,98.9603900000000%2016.7101170000000,98.9603900000000%2026.2178780000000,101.1878500000000%2026.2178780000000,101.1878500000000%2016.7101170000000%20)))%22)&rows=25&start=0) is then parsed in python.



###Known bugs for sending queries:
* Combining queries together using OR gives unexpected results
* Very rare bug (~1/60000) for if the timestamp is exactly on the minute - format becomes incorrect


Returns the name of the matching files and a URL to download from:
* Generally downloads are pretty slow on lx02, and session keeps getting interrupted

### Data processed so far
* All Caliop CLay 1km V4.10 products from April 2018 were downloaded onto lx02 (daytime)
* All files were processed using ~ 9000 requests (1 exception)
* ~ 500 pairings identified in Matches.txt
* So far around 100 SLSTR files downloaded onto lx02 from ESA ([1 exception](https://scihub.copernicus.eu/s3/odata/v1/Products('147bc5af-5478-4268-8763-69f409486d4d')/$value))

### Future processing
We intend to store the data from matching pixels into a new file to generate a dataset which does not require the satellite data.
Variables to be stored:
* Clatitude
* Clongitude
* Ctime
* CFeature_Classification_Flags
* S1 - S9
* Stime
* Slatitude
* Slongitude

Probably using np.arrays, although possible may switch to pandas.dataframe

###Neural Networks
Kenza has written the code to input values from multiple CALIOP and SLSTR files into the 1km CNN (see ccn_1km.py and PrepData.py). She has also written some functions to evaluate future networks (see model_evaluation.py):
* The accuracy
* The accuracy vs number of inputs
* A ROC curve 
* The AUC 
* A precision vs recall curve
* A confusion matrix 

We also discussed meshing alternative network stuctures to make the model more accurate:
* reinforcment learning: un-supervised method, unsuitable as the performance is linked to weather cloud can be directly identified 
* recurrent network, supervised method, allows information to transported backwards and forwards. This could potentially be useful. 

Although before we consider changing the structure we should also try to now evaluate hyperameters, the numbers of layers, the optimiser and activation functions to optimise our current network. 

### Comments 
* For some reason we cannot access the latest V3 files. Permission is denied by NASA.
* Although we are now using a CNN, we think we could exploit it better by encoding more information about the relative positions of the pixels to each (currently taking a 5x5km chunk of SLSTR pixels and dumping them into a list. 

### For next week
*  Complete processing for April
*  plot images to see where the model seems to be failing (snow, coastline, sunglint)
*  use our evaluation tools to determine the hyperameters, the numbers of layers, the optimiser and activation functions to optimise our current network.
*  also use them to make sure there are not any weird correlations.
*  start thinking about how we could use texture instead of a pixel by pixel approach. 
