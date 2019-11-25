import logging as log
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# configuring log
log.basicConfig(filename='log/app.log', level=log.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')


class ExploratoryDataAnalysis():
    def __init__(self):
        log.info("ExploratoryDataAnalysis Created")

    '''
    *****--------------------  EXPLORATORY DATA ANALYSIS FOR ARIMA ---------------------------******

    '''

    def test_stationarity(self, data, col=None):

        '''
        this function runs the Dickey-Fuller test for stationarity
        :return: logs the result of the test
        '''

        log.info("Result of Dickey-Fuller Test")
        dftest = adfuller(data[col], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['statistics', 'p-value', '# lag used', 'No. of Obs used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value %s' % key] = value

        log.info(dfoutput)
        log.info("Dickey-Fuller Test Done")
        return dfoutput

    def data_transformation(self, data):

        '''
        this function transforms the data in log, moving average, log of moving average, differencing of the log,
        differencing of the moving average and exponential weighted moving average and other combinations
        :return: transformed data frame
        '''

        log.info("Data transformation started")
        data['y_log'] = data['y'].apply(lambda x: np.log(x))
        data['y_log_moving_avg'] = data['y_log'].rolling(window=7, center=False).mean()
        data['y_moving_avg'] = data['y'].rolling(window=7, center=False).mean()

        data['y_log_diff'] = data['y_log'].diff()
        data['ts_moving_avg_diff'] = data['y'] - data['y_moving_avg']
        data['ts_log_moving_avg_diff'] = data['y_log'] - data['y_log_moving_avg']
        data_transform = data.dropna()

        # Logged Expnonentially weighted moving average EWMA
        data_transform['y_log_ewma'] = data_transform['y_log'].ewm(halflife=7,
                                                                 ignore_na=False,
                                                                 min_periods=0,
                                                                 adjust=True).mean()
        data_transform['y_log_ewma_diff'] = data_transform['y_log'] - data_transform['y_log_ewma']
        log.info(data_transform.head())
        log.info("Data transformation Done")

        return data_transform

    def get_acf_pacf(self, data):

        '''
        this function gets the auto correlation and partial auto-correlation data
        :return: acf and pacf data
        '''

        log.info("Getting acf and pacf data")
        lag_acf = acf(np.array(data["y_log_diff"]), nlags=20)
        lag_pacf = pacf(np.array(data["y_log_diff"]), nlags=20)
        log.info("acf and pacf data acquired")

        return lag_acf, lag_pacf