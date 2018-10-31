# Material for meeting 02/11/18

Tom and I have trained a model on an image or 7 200 000 pixels of data using a 5 layer feed-forward neural network (see multilayer_peceptron.py). The truth data was taken as the empirical masks provided.

We also tried downloaded data from the CALIPSO instrument and start to colocate it with the some of the satellite images.

Problems:
* the bayes masks only return 0s or 255s everywhere.
* the new version of satpy has some serious bugs when calling the spatial coordinates and the masks.
* the CALIPSO data does not go up to the most recent SENTINEL 3 data.
