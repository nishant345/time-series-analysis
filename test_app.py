import unittest
import pandas as pd
import numpy as np
from configparser import ConfigParser

from application.data_load import DataLoad
from application.initial_data_analysis import InitialDataAnalysis
from application.exploratory_data_analysis import ExploratoryDataAnalysis
from application.data_modelling import DataModelling

class TestDataLoad(unittest.TestCase):

    def _get_config_data(self):

        '''
        function to get configuration data
        :return: configuration data
        '''

        config_data = dict()
        parser = ConfigParser()
        parser.read('config.ini')
        config_data['DATA_STORE_PATH'] = parser.get('general', 'data_store_path')
        return config_data

    def get_data(self):

        '''
        function to get data set
        :return: data set
        '''

        dl = DataLoad()
        config = self._get_config_data()
        data = dl.load_data(config['DATA_STORE_PATH'])
        return data

    def test_load_data(self):
        '''
        test the loading of data
        :return: Nothing
        '''

        data = self.get_data()

        # testing the shape and type of data
        self.assertEqual(len(data), 17379)
        self.assertEqual(len(data.columns), 17)
        self.assertTrue(isinstance(data, pd.DataFrame))
        return

    def test_missing_values(self):

        '''
        test the missing values
        :return:
        '''

        ida = InitialDataAnalysis()
        data = self.get_data()
        missing_values  = ida._get_missing_values(data)

        self.assertEqual(missing_values.any(), 0)
        return

    def test_describe_data(self):

        '''
        test the description of data
        :return:
        '''

        ida = InitialDataAnalysis()
        data = self.get_data()

        self.assertIn('count', data['cnt'].describe())
        self.assertIn('mean', data['cnt'].describe())
        self.assertIn('std', data['cnt'].describe())
        self.assertIn('min', data['cnt'].describe())
        self.assertIn('max', data['cnt'].describe())

    def test_stationarity_raw_data(self):

        '''
        test if data is stationary ;
        if p-value very less data is not stationary
        :return:
        '''

        eda = ExploratoryDataAnalysis()
        data = self.get_data()
        result = eda.test_stationarity(data, 'cnt')
        p_value = result[1]
        self.assertLess(p_value, 0.0000001)

    def test_data_transformation(self):

        '''
        test the transformed data
        only log value for a particular row
        :return:
        '''

        eda = ExploratoryDataAnalysis()
        data = self.get_data()
        df_arima = pd.DataFrame({'ds': data['dteday'], 'y': data['cnt']})
        transformed_data = eda.data_transformation(df_arima)

        self.assertEqual(transformed_data['y_log'][6], np.log(data['cnt'][6]))

    def test_type_acf_pacf(self):

        '''
        test the type of acf and pacf data
        :return:
        '''

        eda = ExploratoryDataAnalysis()
        data = self.get_data()
        df_arima = pd.DataFrame({'ds': data['dteday'], 'y': data['cnt']})
        transformed_data = eda.data_transformation(df_arima)

        acf, pacf = eda.get_acf_pacf(transformed_data)
        self.assertTrue(isinstance(acf, np.ndarray))
        self.assertTrue(isinstance(pacf, np.ndarray))

    def test_ARIMA_modelling(self):

        '''
        test the value of error metric is less than one
        :return:
        '''

        eda = ExploratoryDataAnalysis()
        data = self.get_data()
        df_arima = pd.DataFrame({'ds': data['dteday'], 'y': data['cnt']})
        transformed_data = eda.data_transformation(df_arima)
        dm = DataModelling()

        mae, rmse, result = dm.ARIMA_modelling(transformed_data, col='y_log_diff', p=0, d=0, q=2)
        self.assertLess(mae, 1)


if __name__ == '__main__':
    unittest.main()
