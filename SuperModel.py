
##############################################
# (c) Copyright 2018-2019 Kenza Tazi and Thomas Zhu                                        
# This software is distributed under the terms of the GNU General Public
# Licence version 3 (GPLv3)
##############################################

import FFN
import CNN
import DataPreparation as dp
import ModelApplication as app
import ModelEvaluation as me
import DataLoader as DL
import Visualisation as Vis

class SuperModel():
    def __init__(self, name, FFN1, FFN2, CNN):
        self.name = name
        self.FFN1 = FFN1
        self.FFN2 = FFN2
        self.CNN = CNN

    def predict_file(self, Sreference):

        ftestdata = dp.getinputs(Sreference, num_inputs=24)  # include indices
        indices = np.arange(7200000)

        # exclude indices when predicting
        predictions1 = self.FFN1.Predict(ftestdata)
        augm_data = np.column_stack((predictions1, indices))  # merge with indices

        gooddata = [x for x in augm_data if not(0.4 <= x[0] < 0.6)]
        poordata = [x for x in augm_data if (0.4 <= x[0] < 0.6)]

        ctestdata = dp.context_getinputs(scene, poordata)
        predictions2 = self.CNN.predict(
            ctestdata[0])   # May require .model.predict

        predictions3 = self.FFN2.predict(
            np.column_stack((augm_data[:, 0], predictions2)))

        fixeddata = np.column_stack((predictions3, poordata[:, -1]))

        final_predictions_with_indices = np.concatenate(gooddata, fixeddata)
        final_predictions = (np.sort(final_predictions_with_indices))[:, 0]

        return final_predictions
