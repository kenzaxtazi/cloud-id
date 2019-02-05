import unittest
from ModelApplication import apply_mask
import numpy as np
from ffn2 import FFN

class TestMaskingMethod(unittest.TestCase):

    def test_mask(self):
        Sfilename = "./SatelliteData/SLSTR/2018/08/S3A_SL_1_RBT____20180806T081914_20180806T082214_20180807T131253_0179_034_178_1620_LN2_O_NT_003.SEN3"

        model = FFN('Net1_FFN', 'Network1')
        model.Load()


        mask = apply_mask(model.model, Sfilename, 24)[0]

        self.assertEqual(mask.shape, (2400, 3000))


if __name__ == '__main__':
    unittest.main()
