 
# Material for meeting 18/10/18

We first found that the anaconda files for the satpy library are old. We needed to dowload those from the satpy Github. Tom fixed the files to run on Windows and Kenza will be using her Mac. 
 
We can succesfully dowload the data from CEDA, access the arrays and produce images. The S7, S8, and S9 files channels are not accessible using the satpy SLSTR reader. The error was previously spotted by another Github user. We went into files of the library to fix this using the other users code but it did not work. 

File format e.g.: 
S3A_SL_1_RBT____20180822T000619_20180822T000919_20180822T015223_0179_035_016_3240_SVL_O_NR_003.SEN3

Mission ID | Processing level | Datatype | Start time | End time | Creation time | Duration | Cycle | Relative orbit| Frame | Center | Mode | Timeliness | Collection 
---------- | ---------------- | -------- | ---------- | ---------| ------------- | -------- | ----- | --------------| ----- | ------ | ---- | ---------- | ------------
S3A | 1S | RBT |20180822T000619 | 20180822T000919| 20180822T015223 | 0179 | 035 | 016 | 3240 | SVL | O | NR | 003
Sentinel 3A | Level 1 | ? | 22nd August 2018 @ 00:06:19 UTC | 22nd August 2018 @ 00:09:19 UTC| 22nd August 2018 @ 01:52:23 UTC | 179s | 35 multiples of 385 orbits (385 orbits are completed before the ground tracks are repeated) | 35th orbits in cycle | 16 | Svalbard processing center |  

Tom also found KLM files to plot out the satelite's ground tracks on Fusion Tables or Google Earth. This will help us easily identify which files we want to look out without having to download them. 
https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-3/satellite-description/orbit

Below we have generated an image from a single channel, a false colour image, images overlayed the included cloud masks (empirical and bayesian), and images overlayed with only one of the bits for either the cloud mask. The coressponding code can be found here under Plotting_images_and_masks.py . 

![pic1](/Images/S1_n.png)
Figure 1: Northeastern Australia using channel S1

![pic2](/Images/nothernaustralia_falsecolour.png)
Figure 2: False colour image of Northeastern Australia using channel S1, S2, S3 (S1=blue, S2=green, S3=red)

![pic3](/Images/S1n_S3A_SL_1_RBT____20180404T190108_LN2_O_NT_003-(10.442133,-142.862277).png)
Figure 3: Pacific Ocean using channel S1.

![pic4](/Images/Figure_1-1.png)
Figure 4: False colour image of Pacific Ocean using above technique

![pic5](/Images/Figure_2.png)
Figure 5: Pacific Ocean with summed bits from empirical cloud mask.

![pic6](/Images/bitmasks.png)
Figure 6: Pacific Ocean with various bits from empirical cloud mask. 


Comments: 
- many of the pictures are very poor and have lots of nan's.
- Bayesian cloud mask fails completely for the Pacific Ocean example.

Questions:
- why are there no b channels for S1-3?
- what do the missing cells in the table mean? 
- should we fix the S7-9 channel reading?

Work for next meeting: 
- Set up shop on cluster 
- Calculate cloud masks for a collection of data 
- Read about empirical tecniques 
- Start thinking about how to set up the neural network

