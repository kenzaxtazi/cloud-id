# Material for meeting on 21/11/18

We are succesfully training our network on CALIOP data. 

Tom has streamlined to dowload and collocation function to make less calls and dowloaded the relevant data en masse. 

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
*  plot images to see where the model seems to be failing (snow, coastline, sunglint)
*  use our evaluation tools to determine the hyperameters, the numbers of layers, the optimiser and activation functions to optimise our current network.
*  also use them to make sure there are not any weird correlations.
*  start thinking about how we could use texture instead of a pixel by pixel approach. 
