## Material for meeting 29/11/18

### Updates from last week
Data has been downloaded and matching has been done on a pixel by pixel basis. 
This information is stored as a pandas dataframe with 28 fields of raw data. 
For the first stage of processing, 2 additional columns were calculated, the time separation (and the order of satellite arrival) and the distance between pixel locations (using geodesic formula).

Approximately 3 months of data has been processed so far, currently storing data on around 750,000 collocated pixels in pickle file format ~ 200MB. This comes from ~ 1 TB of raw data. 

#### Data processing workflow

1. Calipso data is downloaded from https://search.earthdata.nasa.gov/ ~16GB per month (day files only).
2. Matches are found using ESA queries ~ 1 hour per month of data. The matches are stored in .txt format.
3. The identified SLSTR files are downloaded. This is the slowest part of the process ~ 7 hours
  * ESA downloads are reliable, typically 1 in 200 files fail to download, speeds are typically ~ 10MBps (probably due to Sentinel3 data not being on the main server).
  * CEDA FTP downloads are reliable if the files are present. More recent files are missing. Download speeds are typically ~ 20MBps. 
  * CEDA http downloads should be the most reliable. Uses OpenID authentication which requires creation of temporary certificates. Test run was successful using cURL at ~ 10MBps.
4. Once all files are downloaded, the collocation is performed on the paired files and the relevent pixels are stored. This takes ~ 1 hour and produces the output .pkl files.

#### Known issues
* The time difference calculated is only correct +- 90 seconds as we treat all pixels in a single SLSTR file as being contemporaneous. It should be possible to use linear interpolation to make better time measurements of SLSTR pixels.
* Strange artifcats on some [images](http://www.hep.ph.ic.ac.uk/~trz15/Figure_7.png) 
* The acceptable time window seems to be only a few minutes, large time differences lead to poor truthing
* For the most recent files, we have been looking at S3B data. Some of the channel values are negative, these seem to correlate with [dead pixels](http://www.hep.ph.ic.ac.uk/~trz15/Figure_11.png).
* All data are from polar regions, latitudes between 65.20 and 81.16 degrees. However, as we are selecting day files, in certain months we only have data from either the north or the south.
* Depending on the month, we are using slightly different Calipso products.
* Running some of the scripts requires setting the environment variable OMP_NUM_THREADS=1, this seems to be related to the tqdm module we are using as a progress bar. There is no noticeable difference in performance when this is set.
* Using screen produces this [message](https://imgur.com/a/EyocbU2) on startup. Also requires deactivating and reactivating the virtual environment to access packages despite which python being set correctly.
* Of the pairings identified in stage 2., around 1 in 10 fail the first step of the collocation function.


### Comments
* The scripts in use should scale easily, we should be able to process the data for different months in parallel. Currently using screen to be able to detach from session.

We are running the data data TOm dowloaded into 6 layer feed-forward network with the 9 channels as inputs. We first tried to vary the threshhold for the time difference between the CALIOP and SLSTR pixels. The results are shown in the graph below. The accuracy of the network peaks at about 250s. Below this limit, the model probably doesn't have enough data learn adaquately (<50'000 points). Above this limit, the correlation between the cloud presence is diminished. I think the variations probably come from the fact that more time does not mean less correlation. For e.g. strong winds could make the  scene change drastically.

<img src=/Images/Time_difference_vs_accuracy2.png width="600">
Fig 1: Accuracy as a function of time difference. 

&nbsp;  

For a threshold of 250s we looked at the model evaulations I wrote for last time. The accuracy is 84%, area under the ROC curve is 0.91. Below are the graphs for the ROC curve, Precision vs. Recall curve and the table for the confusion matrix.


<img src=/Images/ROC.png width="800">
Fig 2: ROC Curve for validation set (500 random points).

&nbsp; 

<img src=/Images/PvR.png width="800"> 
Fig 3: Precicion as a function of recall for validation set.

&nbsp;

Truth | Predicted as Not Cloud | Predicted as Cloud
------------ | -------------| ----------
Cloud | 38 | 253
Not Cloud | 156 | 53

Table I: Confusion matrix of validation set. 

We repeated this process for the same time threshold but with dowloading almost twice as much data (which took over 6 hours to complete because the files wer non-existent on CEDA). The accuracy was 73%, area under the ROC curve is 0.79. Below are the graphs for the ROC curve, Precision vs. Recall curve and the table for the confusion matrix.

<img src=/Images/ROClarge.png width="800"> 
Fig 4: ROC Curve for larger training set.

&nbsp;

<img src=/Images/PvRlarge.png width="800"> 
Fig 5: Precicion as a function of recall for larger training set.

&nbsp;

Truth | Predicted as Not Cloud | Predicted as Cloud
------------ | -------------| ----------
Cloud | 20 | 319 
Not Cloud | 48 | 113

Table II: Confusion matrix for larger training set.

&nbsp;

We also plotted the some test data (100 new points for a single test image) as a sanity check but also to see where the classifier was struglling. But this is prooving hard to do without an interactive environment 

<img src=/Images/ploooot.png width="800">  
Fig 6: Test image 

&nbsp;

### Comments:
* we were suprised the accuracy decreased with the increased number of inputs

### Next week: 
* add ancilliary data to the network 





