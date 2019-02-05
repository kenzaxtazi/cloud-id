import DataLoader as DL
import DataPreparation as dp
import numpy as np


class SuperModel():
    def __init__(self, name, FFN1, FFN2, CNN):
        self.name = name
        self.FFN1 = FFN1
        self.FFN2 = FFN2
        self.CNN = CNN
    
    def predict_file(self, Sreference):

        scene = DL.scene_loader(Sreference)

        ftestdata = dp.getinputs(scene, num_inputs=24, indices=True) #include indices
        indices = ftestdata[-1]

        predictions1 = self.FFN1.Predict(ftestdata[:-1]) #exclude indices when predicting 
        augm_data = np.column_stack((predictions1, indices)) # merge with indices

        gooddata = [x for x in augm_data if not(0.4 <=x[0] <0.6)]
        poordata = augm_data[0.4 <= augm_data[:, 0] < 0.6]
        
        ctestdata = dp.context_getinputs(scene, poordata)  
        predictions2 = self.CNN.predict(ctestdata[0])   # May require .model.predict

        predictions3 = self.FFN2.predict(
            np.column_stack((augm_data[:,0], predictions2)))

        fixeddata= np.column_stack((predictions3, poordata[:,-1]))

        final_predictions_with_indices = np.concatenate(gooddata, fixeddata)
        final_predictions = (np.sort(final_predictions_with_indices))[:,0]

        return final_predictions
