import unittest

from Collocation import collocate


class TestCollocationMethod(unittest.TestCase):

    def test_collocate(self):
        Cfilename = "./SatelliteData/Calipso/Calipso1km/2018/08/CAL_LID_L2_01kmCLay-Standard-V4-20.2018-08-06T07-39-02ZD.hdf"
        Sfilename = "./SatelliteData/SLSTR/2018/08/S3A_SL_1_RBT____20180806T081914_20180806T082214_20180807T131253_0179_034_178_1620_LN2_O_NT_003.SEN3"

        coords = collocate(Sfilename, Cfilename)
        num_pixels = len(coords)
        self.assertEqual(num_pixels, 1256)


if __name__ == '__main__':
    unittest.main()
