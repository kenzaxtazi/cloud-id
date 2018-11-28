## Material for meeting 29/11/18

We are running the data data TOm dowloaded into 6 layer feed-forward network with the 9 channels as inputs. We first tried to vary the threshhold for the time difference between the CALIOP and SLSTR pixels. The results are shown in the graph below. The accuracy of the network peaks at about 250s. Below this limit, the model probably doesn't have enough data learn adaquately (<50'000 points). Above this limit, the correlation between the cloud presence is diminished. I think the variations probably come from the fact that more time does not mean less correlation. For e.g. strong winds could make the  scene change drastically.

Fig 1: Accuracy as a function of time difference. 

For a threshold of 250s we looked at the model evaulations I wrote for last time. The accuracy is 84%, area under the ROC curve is _ _ _ . Below are the graphs for the ROC curve and  the table for the confusion matrix.

Fig 2: ROC Curve for validation set (500 random points).

Table I: Confusion matrix of validation set. 

We repeated this process for the same time threshold but with dowloading almost twice as much data (which took over 6 hours to complete because the files wer non-existent on CEDA).

Fig 3: ROC Curve for larger training set.

Table II: Confusion matrix for larger training set

We also plotted the some test data (100 new points for a single test image) as a sanity check but also to see where the classifier was struglling.


Fig 4: Test image 







