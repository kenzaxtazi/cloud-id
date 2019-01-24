# Material for meeting 02/11/18

Tom and I have trained a model on an image or 7 200 000 pixels of data (although it stabilises at 81% accuracy at around 100 000 pixels) using a 5 layer feed-forward neural network (see multilayer_peceptron.py). The truth data was taken as the empirical masks provided.


Functions have been written to handle data for SLSTR measurements from CEDA's FTP server. Using the tool at https://scihub.copernicus.eu/s3/#/home it is simple to find appropriate relative orbit number and framenumber for any point on the Earth. Given these values, functions inside DataLoader.py are able to use a regex to find all the available files that match going back to 2016. The regex can be further adapted to search based on any criteria in the file naming convention such as date/time. Some anomalies were found with the files available on the FTP server; e.g folders for some days did not exist and read permissions were not available for individual files. 

Nonetheless, we are now able to find relevent files, normally around 12 (24 since one from each processing centre) for any view of the Earth. We can then download them onto our own computers or the HEP machines. A small amount of additional work is required if emulating the original file structure in the shared folder is desired.

We also tried to download data from the CALIPSO instrument and start to colocate it with the some of the satellite images.

Problems:
* the bayes masks only return 0s or 255s everywhere for all ~40 files tested.
* the new version of satpy has some serious bugs when calling the spatial coordinates and the masks.
* Currently have two versions of code since the earlier mentioned pull request was merged with the master satpy branch. The new nc_slstr reader now uses a different naming convention.
* the CALIPSO data does not go up to the most recent SENTINEL 3 data.
