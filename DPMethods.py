import pandas as pd
import DataPreparation as dp
import numpy as np

# Add extra methods to DataFrame to allow simple data processing


@pd.api.extensions.register_dataframe_accessor("dp")
class DataPreparer():
    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self._obj = pandas_obj

    def remove_nan(self):
        if 'confidence_in' in self._obj.columns:
            self._obj = self._obj.drop(['confidence_in'], axis=1)
        self._obj = self._obj.dropna()
        return(self._obj)

    def remove_anomalous(self, MaxDist=500, MaxTime=1200):
        self._obj = self._obj[self._obj['Distance'] < MaxDist]
        self._obj = self._obj[abs(self._obj['TimeDiff']) < MaxTime]
        return(self._obj)

    def shuffle_random(self, validation_frac=0.15):
        self._obj = self._obj.sample(frac=1)
        return(self._obj)

    def shuffle_by_file(self, validation_frac=0.15):
        Sfiles = list(set(self._obj['Sfilename']))
        np.random.shuffle(Sfiles)
        sorterindex = dict(zip(Sfiles, range(len(Sfiles))))
        self._obj['Temp'] = self._obj['Sfilename'].map(sorterindex)
        self._obj = self._obj.sort_values(['Temp'])
        self._obj = self._obj.drop(['Temp'], axis=1)

    def get_training_data(self, input_type=24, validation_frac=0.15):
        self.remove_nan()
        self.remove_anomalous()
        self.shuffle_by_file(validation_frac)

        pixel_channels = (self._obj[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in',
                                     'satellite_zenith_angle', 'solar_zenith_angle', 'latitude_an', 'longitude_an']].values).astype(float)
        confidence_ints = self._obj['confidence_an'].values

        confidence_flags = dp.bits_from_int(confidence_ints, input_type)

        confidence_flags = confidence_flags.T

        pixel_indices = self._obj.index.values

        pixel_inputs = np.column_stack((pixel_channels, confidence_flags))

        pixel_outputs = self._obj[[
            'Feature_Classification_Flags', 'bayes_in', 'cloud_an', 'TimeDiff']].values

        pix = np.column_stack((pixel_inputs, pixel_outputs))
        pix = np.column_stack((pix, pixel_indices))

        pix = pix.astype('float')

        pct = int(len(pix)*validation_frac)
        training = pix[:-pct, :]   # take all but the 15% last
        validation = pix[-pct:, :]   # take the last 15% of pixels

        training_data = training[:, :input_type]
        training_truth_flags = training[:, input_type]
        validation_data = validation[:, :input_type]
        validation_truth_flags = validation[:, input_type]

        training_cloudtruth = (training_truth_flags.astype(int) & 2) / 2
        reverse_training_cloudtruth = 1 - training_cloudtruth
        training_truth = np.vstack(
            (training_cloudtruth, reverse_training_cloudtruth)).T

        validation_cloudtruth = (validation_truth_flags.astype(int) & 2) / 2
        reverse_validation_cloudtruth = 1 - validation_cloudtruth
        validation_truth = np.vstack(
            (validation_cloudtruth, reverse_validation_cloudtruth)).T

        return_list = [training_data, validation_data, training_truth,
                       validation_truth]
        return return_list

    def get_inputs(self, input_type=24):

        pixel_channels = (self._obj[['S1_an', 'S2_an', 'S3_an', 'S4_an', 'S5_an', 'S6_an', 'S7_in', 'S8_in', 'S9_in',
                                     'satellite_zenith_angle', 'solar_zenith_angle', 'latitude_an', 'longitude_an']].values).astype(float)

        confidence_ints = self._obj['confidence_an'].values

        confidence_flags = dp.bits_from_int(confidence_ints, input_type)

        confidence_flags = confidence_flags.T

        pixel_inputs = np.column_stack((pixel_channels, confidence_flags))

        inputs = np.vstack((pixel_inputs, confidence_flags))

        return(inputs.T)

    def make_attrib_hist(self, column='latitude_an'):
        out = self._obj[column]
        frq, edges = np.histogram(out, 100)
        plt.title(column + ' histogram')
        plt.bar(edges[:-1], frq, width=np.diff(edges), ec='k', align='edge')
        plt.show()

    def make_CTruth_col(self):
        FCF = self._obj['Feature_Classification_Flags']
        val = FCF.astype(int)
        val = val & 7
        CTruth = val == 2
        self._obj['CTruth'] = pd.Series(CTruth, index=df.index)
        return(self._obj)

    def make_STruth_col(self, cloudmask='cloud_an', bit=1):
        bitfield = self._obj[cloudmask]
        val = bitfield.astype(int)
        val = val & bit
        STruth = val == bit
        self._obj['STruth'] = pd.Series(STruth, index=df.index)
        return(self._obj)
