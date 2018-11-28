## Material for meeting 29/11/18

We are running the data data TOm dowloaded into 6 layer feed-forward network with the 9 channels as inputs. We first tried to vary the threshhold for the time difference between the CALIOP and SLSTR pixels. The results are shown in the graph below. The accuracy of the network peaks at about 250s. Below this limit, the model probably doesn't have enough data learn adaquately (<50'000 points). Above this limit, the correlation between the cloud presence is diminished. I think the variations probably come from the fact that more time does not mean less correlation. For e.g. strong winds could make the  scene change drastically.

Fig 1. Accuracy as a function of time difference. 

For a threshold of 250s we looked at the model evaulations I wrote for last time. The accuracy is 84%, area under the ROC curve is ___. Below are the graphs for the ROC curve and  the table for the matrix correlation. 

Fig 2. 

Table I.
