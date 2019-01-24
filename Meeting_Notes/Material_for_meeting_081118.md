# Material for meeting 08/11/18

This week we:
* Opened CALIPSO data files
* Created multidimensional CNN made for the 5x5km CALIPSO products (see first_cnn.py) 
* Tried to colocate and to retrieve optical depth values from CALIPSO files to creat truth data set 

Calipso Files:
Data available from https://search.earthdata.nasa.gov.

Products highlighted by Dr Poulsen:

* CAL_LID_L2_01kmCLay-Standard-V4-10
* CAL_LID_L2_05kmCLay-Standard-V4-10
* CAL_LID_L2_05kmALay-Standard-V4-10
* CAL_LID_L2_VFM-Standard-V4-10

All available from 2006 until 2 June 2018 with some dates (e.g. 20 May 2018) missing. The system requires ordering granules which are then processed and made available on an FTP server. Examples for the first two products from 1 April 2018 have been downloaded, copies of which are on cloud directory of lx02 ~18GB per month. We have emailed the Earthdata Search Support to find out when more recent files will be released

Files are in HDF4 format. Satpy does not support these files properly. The only relevent submodule is "satpy.readers.hdf4_caliopv3" however it is only partially developed. The .yaml file at https://github.com/pytroll/satpy/blob/master/satpy/etc/readers/hdf4_caliopv3.yaml has a regex which only matches night time files and only extracts four variables, one of which does not appear in the headers of the downloaded files. The first product has 73 variables whilst the second has 174. The variable headers have been noted in the Git repo. The two variables of interest identified in the 5km product by Dr Poulsen, "SD_FEATURE_CLASSIFICATION_FLAGS" and "SD_FEATURE_OPTICAL_DEPTH_532" have been downloaded but require additional processing. The former requires porting a script written in IDL at https://eosweb.larc.nasa.gov/sites/default/files/project/calipso/tools/vfm_feature_flags.pro and the latter has a strange output (different number of values in each entry). Additional variables of interest include "Number_Layers_Found" which returns an integer (observed between 0 and 7). Its description in the documentation at https://eosweb.larc.nasa.gov/PRODOCS/calipso/Quality_Summaries/CALIOP_L2LayerProducts_2.01.html is:

"Number Layers Found (provisional)
The number of layers found in this column; cloud data products report (only) the number of cloud layers found, and aerosol report
(only) the number of aerosol layers found."

Its distribution is shown in Image [pic2].

![pic2](https://imperiallondon-my.sharepoint.com/personal/kt2015_ic_ac_uk/Documents/Forms/All.aspx?slrid=ad28b99e%2D2021%2D7000%2D8f31%2Da649b2b7038f&RootFolder=%2Fpersonal%2Fkt2015%5Fic%5Fac%5Fuk%2FDocuments%2FImages&FolderCTID=0x0120008872E2E669FB2044B088F3F3E5CCF65B#slrid=ad28b99e%2D2021%2D7000%2D8f31%2Da649b2b7038f&FolderCTID=0x0120008872E2E669FB2044B088F3F3E5CCF65B&id=%2Fpersonal%2Fkt2015%5Fic%5Fac%5Fuk%2FDocuments%2FImages%2FNumLayers%2Epng&parent=%2Fpersonal%2Fkt2015%5Fic%5Fac%5Fuk%2FDocuments%2FImages)

Currently using a dedicated program to read the files https://portal.hdfgroup.org/display/support/HDF+4.2.14 . Variables are referenced by their index and loaded as a string into python. The string format changes between products and variables. The string is then manipulated to form numpy arrays. In the second product type, the string implies that the data is in an (n, 3) shaped array. However, if this array is flattened, the time (Given in IAT), latitude and longitude are monotically increasing which implies that it is actually an (n, 1) shaped array.

Started to use CIS http://cistools.net/ to perform collocation. It uses python packages to read the HDF4 files which means that using the dedicated program is unnecessary. Installation is difficult as the Iris dependency is not working as expected. Successful installation requires manually using the latest files for CIS, Iris, udunits2, netcdf, netcdftime and cf-units and editing an import statement in one of the packages. 

CIS does not directly support the four products above, only CAL_LID_L2_05kmAPro. However the code can probably be adapted to accept the relevent ones. For initial testing, a few of the supported products were downloaded. CIS was used to produce basic plots on a map. Attempts were made at using the collocation function. The collocation function was run using default settings, however it failed to return anything after more than an hour of running.

![pic1](https://imperiallondon-my.sharepoint.com/personal/kt2015_ic_ac_uk/Documents/Forms/All.aspx?slrid=ad28b99e%2D2021%2D7000%2D8f31%2Da649b2b7038f&RootFolder=%2Fpersonal%2Fkt2015%5Fic%5Fac%5Fuk%2FDocuments%2FImages&FolderCTID=0x0120008872E2E669FB2044B088F3F3E5CCF65B#slrid=ad28b99e%2D2021%2D7000%2D8f31%2Da649b2b7038f&FolderCTID=0x0120008872E2E669FB2044B088F3F3E5CCF65B&id=%2Fpersonal%2Fkt2015%5Fic%5Fac%5Fuk%2FDocuments%2FImages%2FCISOUT1%2Epng&parent=%2Fpersonal%2Fkt2015%5Fic%5Fac%5Fuk%2FDocuments%2FImages)
