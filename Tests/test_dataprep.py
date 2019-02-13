import unittest
import DataPreparation as DP


class TestInputPrepMethod(unittest.TestCase):

    def test_13inputs(self):
        Sfilename = "./SatelliteData/SLSTR/2018/08/S3A_SL_1_RBT____20180806T081914_20180806T082214_20180807T131253_0179_034_178_1620_LN2_O_NT_003.SEN3"
        inputs = DP.getinputs(Sfilename, 13)
        self.assertEqual(inputs.shape, (7200000, 13))

    def test_22inputs(self):
        Sfilename = "./SatelliteData/SLSTR/2018/08/S3A_SL_1_RBT____20180806T081914_20180806T082214_20180807T131253_0179_034_178_1620_LN2_O_NT_003.SEN3"
        inputs = DP.getinputs(Sfilename, 22)
        self.assertEqual(inputs.shape, (7200000, 22))

    def test_24inputs(self):
        Sfilename = "./SatelliteData/SLSTR/2018/08/S3A_SL_1_RBT____20180806T081914_20180806T082214_20180807T131253_0179_034_178_1620_LN2_O_NT_003.SEN3"
        inputs = DP.getinputs(Sfilename, 24)
        self.assertEqual(inputs.shape, (7200000, 24))


if __name__ == '__main__':
    unittest.main()
