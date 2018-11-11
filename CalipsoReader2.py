# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:03:33 2018

@author: tomzh
"""

from pyhdf.SD import SD, SDC

filename = "D:/SatelliteData/Calipso1km/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T00-04-48ZD.hdf"

file = SD(filename, SDC.READ)



    
def load_hdf(filename):
    """Loads the hdf4 object into memory"""
    file = SD(filename, SDC.READ)
    return(file)
    
def get_header_names(file):
    """Print the names of the dataset names"""
    datasets_dic = file.datasets()
    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)
        
def load_data(file, variable):
    """From the file, load the chosen variable. Valid options in get_header_names()"""
    sds_obj = file.select(variable)
    data = sds_obj.get()
    return(data)
    
if __name__ == '__main__':
    lat = load_data(file, 'Latitude')
    lon = load_data(file, 'Longitude')
    time = load_data(file, 'Profile_Time')
    utctime = load_data(file, 'Profile_UTC_Time')