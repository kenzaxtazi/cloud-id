# Material for meeting 15/11/18


Using the shell commands (detailed [here](https://scihub.copernicus.eu/userguide/BatchScripting)), Tom wrote two functions to collocate Sentinel 3 images from a CALIOP file. The time range, over which matches our sought, can be changed.
Currently the bash script is sending a https request and formatting the .XML which is returned. Additional options are available which allow the user to begin a download of the relevent data from the ESA server using wget.

It's possible to write python script to send the https request without using the .sh script, however we're unclear whether the benefit is worth it.

Using the first 1km Calipso granule, we were able to identify a region which overlapped with an SLSTR file within 30 - 60 minutes.
SLSTR time: 012743 - 013043
Calipso time: 004828 - 005223  ** (+ 1 hour if using IAT converted time)

![fig1](/Images/Collocation1.png)
Figure 1: Overlap region, Latitude/Longitude Plot. Blue = Calipso measurements, Orange = SLSTR measurements


Once two overlapping files are loaded into Python, we can identify the pixels which correspond. Currently this is done by checking whether any pixels are within 250m latitude and 250m longitude of each other. Processing for each pair of files takes approximately 10 seconds.
![fig2](/Images/FullCollcation1.png)
Figure 2: Elements of the 2D SLSTR data array which correspond to Calipso measurements

![fig3](/Images/FullCollocation2.png)
Figure 3: Satellite position corresponding to collocated pixels, Latitude/Longitude Plot.


For now this has been performed on personal machines. We are unable to load hdf4 files properly as it appears the requisite package has dependencies we cannot install.
https://hdfeos.org/software/pyhdf.php


Kenza translated the IDL file to translate the bitwise information into one hot encoding. She also fixed the CNN to for the 1km truth inputs. 


![fig4](/Images/colourful_collocation.png)
Figure 4: CALIOP measurements superimposed on SLSTR S1 image (0 = no value, 1 = clear, 2 = cloud, 3 = aerosol, 4= stratospheric aerosol, 5 = surface, 6 = subsurface, 7 = undetermined) 


Questions:
* The flags 10 values per 'measurement', we are only taking the first one but we are uncertain if this is the right approach.  
* Are we using V3.40 for the CALIOP data now?
* Should we put the aerosol, stratospheric aerosol and cloud together? Or should we have three categories?
* What are we going to write in our progress report?
* Has Prof Egede merged the nc_slstr.yaml file to the satpy master branch?
* What's the best way to query the SLSTR database, number of queries / query content?


Next week: 
* Download more data to train the model properly
* Plot validtion results to understand what the CNN is failing to pick up

