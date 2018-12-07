# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:03:33 2018

@author: tomzh
"""

from pyhdf.SD import SD, SDC


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


class SDopener():
    # Class to call when using context manager
    def __init__(self, path, mode=SDC.READ):
        self.path = path
        self.mode = mode

    def __enter__(self):
        self.SD = SD(self.path, self.mode)
        return(self.SD)

    def __exit__(self, type, value, traceback):
        self.SD.__del__()


if __name__ == '__main__':
    Cfilename = "D:/SatelliteData/Calipso1km/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T08-19-14ZD.hdf"
    with SDopener(Cfilename) as file:
        clat = load_data(file, 'Latitude')
        clon = load_data(file, 'Longitude')
        ctime = load_data(file, 'Profile_Time')
        ctime += 725846400.0    # Time in UNIX
        ctime -= 10  # Leap second correction
        cutctime = load_data(file, 'Profile_UTC_Time')
