import unittest
from ModelApplication import apply_mask
import numpy as np
from ffn2 import FFN

class TestMaskingMethod(unittest.TestCase):

    def test_mask(self):
        Sfilename = "./SatelliteData/SLSTR/PacificTest/S3A_SL_1_RBT____20171120T190102_20171120T190402_20171122T003854_0179_024_341_2880_LN2_O_NT_002.SEN3"

        model = FFN('Net1FFN', 'Network1')
        model.Load()


        mask = apply_mask(model.model, Sfilename, 24)[0]

        self.assertEqual(mask.shape, (2400, 3000))


if __name__ == '__main__':
    unittest.main()
