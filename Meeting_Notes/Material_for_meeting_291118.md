## Material for meeting 29/11/18

We are running the data data TOm dowloaded into 6 layer feed-forward network with the 9 channels as inputs. We first tried to vary the threshhold for the time difference between the CALIOP and SLSTR pixels. The results are shown in the graph below. The accuracy of the network peaks at about 250s. Below this limit, the model probably doesn't have enough data learn adaquately (<50'000 points). Above this limit, the correlation between the cloud presence is diminished. I think the variations probably come from the fact that more time does not mean less correlation. For e.g. strong winds could make the  scene change drastically.

<img src=/Images/Time_difference_vs_accuracy2.png width="800">
Fig 1: Accuracy as a function of time difference. 

For a threshold of 250s we looked at the model evaulations I wrote for last time. The accuracy is 84%, area under the ROC curve is 0.91. Below are the graphs for the ROC curve, Precision vs. Recall curve and the table for the confusion matrix.

<img src=/Images/ROC.png width="800">
Fig 2: ROC Curve for validation set (500 random points).

<img src=/Images/PvR.png width="800"> 
Fig 3: Precicion as a function of recall for validation set.

Truth | Predicted as Not Cloud | Predicted as Cloud
------------ | -------------| ----------
Cloud | 38 | 253
Not Cloud | 156 | 53

Table I: Confusion matrix of validation set. 

We repeated this process for the same time threshold but with dowloading almost twice as much data (which took over 6 hours to complete because the files wer non-existent on CEDA). The accuracy was 73%, area under the ROC curve is 0.79. Below are the graphs for the ROC curve, Precision vs. Recall curve and the table for the confusion matrix.

<img src=/Images/ROClarge.png width="800"> 
Fig 4: ROC Curve for larger training set.

<img src=/Images/PvRlarge.png width="800"> 
Fig 5: Precicion as a function of recall for larger training set.

Truth | Predicted as Not Cloud | Predicted as Cloud
------------ | -------------| ----------
Cloud | 20 | 319 
Not Cloud | 48 | 113

Table II: Confusion matrix for larger training set.

We also plotted the some test data (100 new points for a single test image) as a sanity check but also to see where the classifier was struglling. But this is prooving hard to do without an interactive environment 

<img src=/Images/testimage.png width="800">  
Fig 6: Test image 


### Comments:
* we were suprised the accuracy decreased with the increased number of inputs

### Next week: 
* add ancilliary data to the network 





