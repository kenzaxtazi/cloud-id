import unittest
from Collocation2 import find_SLSTR_data
import os


class TestQueryMethod(unittest.TestCase):

    def test_esa_query(self):
        Cfilename = "./SatelliteData/Calipso/Calipso1km/2018/04/CAL_LID_L2_01kmCLay-Standard-V4-10.2018-04-01T00-04-48ZD.hdf"
        out = find_SLSTR_data(Cfilename, 120, 20)
        out_Sfiles = out[0]
        out_Surls = out[1]
        expected_Sfiles = ['S3A_SL_1_RBT____20180331T234644_20180331T234944_20180402T040830_0179_029_287_1620_LN2_O_NT_002',
                           'S3A_SL_1_RBT____20180401T012743_20180401T013043_20180402T055007_0179_029_288_1620_LN2_O_NT_002',
                           'S3A_SL_1_RBT____20180331T234944_20180331T235244_20180402T040937_0179_029_287_1800_LN2_O_NT_002',
                           'S3A_SL_1_RBT____20180401T013043_20180401T013343_20180402T055116_0179_029_288_1800_LN2_O_NT_002']
        expected_urls = ["https://scihub.copernicus.eu/s3/odata/v1/Products('f3a54b0c-8de1-413d-b7b8-f94fdbb80a41')/$value",
                         "https://scihub.copernicus.eu/s3/odata/v1/Products('eaa28685-e5fe-4da5-a614-4c4e6938b071')/$value",
                         "https://scihub.copernicus.eu/s3/odata/v1/Products('8d212383-3d08-418d-8011-98c9a818d06d')/$value",
                         "https://scihub.copernicus.eu/s3/odata/v1/Products('9ea22b9e-2296-412d-a2d4-0320b57d70e1')/$value"]
        expected_Sfiles.sort()
        expected_urls.sort()
        out_Sfiles.sort()
        out_Surls.sort()
        self.assertEqual(out_Sfiles, expected_Sfiles)
        self.assertEqual(out_Surls, expected_urls)


if __name__ == '__main__':
    unittest.main()
