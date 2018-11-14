# Material for meeting 15/11/18

Using the shell commands (detailed [here](https://l.messenger.com/l.php?u=http%3A%2F%2Fscihub.copernicus.eu%2Fuserguide%2FBatchScripting&h=AT2X1T7nqJLXnZ9--xG5jxRlvOEfa7dZdcq9kq8pbLriWdWmt4tJ-VQgCnK8r8msAqNrg_DtTHWdnmg57aO4D75v7J1sEwh-4Y3EP9o4_s1JSQoRoZvcdTwc2i8GeUE1uswsRtvoJvQ)), Tom wrote two functions to collocate Sentinel 3 images from a CALIOP file. The time range, over which matches our sought, can be changed.

Kenza translated the IDL file to translate the bitwise information into one hot encoding. She also fixed the CNN to for the 1km truth inputs. 

Questions:
* Are we using V3.40 now?
* What are we going to write in our progress report?
* Has Prof Egede merged the nc_slstr.yaml file to the satpy master branch?

Next week: 
* Download more data to train the model properly
* Plot validtion results to understand what the CNN is failing to pick up

