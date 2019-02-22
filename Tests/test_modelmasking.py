import unittest

from FFN import FFN


class TestMaskingMethod(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.model = FFN('Net1_FFN')
        self.model.Load()
        self.TestFile = "./SatelliteData/SLSTR/2018/08/S3A_SL_1_RBT____20180806T081914_20180806T082214_20180807T131253_0179_034_178_1620_LN2_O_NT_003.SEN3"

    def test_mask_FFNMethod(self):
        mask = self.model.apply_mask(self.TestFile)[0]

        self.assertEqual(mask.shape, (2400, 3000))


if __name__ == '__main__':
    unittest.main()
