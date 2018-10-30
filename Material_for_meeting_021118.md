# Material for meeting 02/11/18

Tom and I have trained a model on an image or 7200 pixels of data using a 5 layer feed-forward neural network (see multilayer_peceptron.py). The truth data was taken as the empirical masks provided.

We have also downloaded data from the CALIPSO instrument and a starting to colocate it with the some of the satellite images.


Problems:
* the bayes masks only return 0s
* the new version of satpy has some serious bugs when calling the spatial coordinates and the masks

Questions:
* Why do the longitude and latitude differ between bands in the latest satpy version?
* 

