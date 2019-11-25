import logging as log
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# configuring log
log.basicConfig(filename='log/app.log', level=log.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')

class DataModelling():
    def __init__(self):
        log.info("DataModelling Object Created")

    def LSTM_Modelling(self, trainX, trainY, testX, testY, params):

        '''
        Machine Learning Modelling
        :return: trainPredict, testPredict
        '''

        log.info("Initializing LSTM Data Modelling ")
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, params["lookback"])))
        model.add(Dense(1))
        model.compile(loss=params['loss'], optimizer=params['optimizer'])
        model.fit(trainX, trainY, epochs=params['epochs'], batch_size=params['batch_size'], verbose=params['verbose'])

        log.info("Making predictions for LSTM Data Modelling ")
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        log.info("LSTM Data Modelling Done")
        return trainPredict, testPredict


    def ARIMA_modelling(self, data, col, p, d, q):

        '''
        Statistical Modelling
        this function is used to model data based on ARIMA concepts
        :return: error metric evaluation (MAE) and prediction
        '''

        log.info("Initializing ARIMA Data Modelling ")
        model = ARIMA(data[col], order=(p, d, q))
        result = model.fit(disp=-1)

        log.info("Making predictions for ARIMA Data Modelling ")
        len_result = len(result.fittedvalues)
        mae = mean_absolute_error(result.fittedvalues, data[col])
        rmse = np.sqrt(mean_squared_error(result.fittedvalues, data[col]))
        '''
        plt.plot(data[col])
        plt.plot(result.fittedvalues, color="red")
        plt.title("for ARIMA model %i %i %i for %s, RSS: %.4f RMSE: %.4f" % (p, d, q, col, rss, rmse))
        plt.show()
        '''
        log.info("ARIMA Data Modelling Done")
        return mae, rmse, result


    def run_sarimax_model(self, data, col, p, d, q):

        '''
        Statistical Modelling taking Seasonality into account
        this function is used to model data based on SARIMAX concepts
        :return: error metric evaluation (MAE)
        '''

        log.info("Initializing SARIMAX Data Modelling")
        model = SARIMAX(data[col], order=(p, d, q), seasonal_order=(4, 1, 4, 12))
        result = model.fit(disp=-1)
        len_result = len(result.fittedvalues)
        mae = mean_absolute_error(result.fittedvalues, data[col])
        rmse = np.sqrt(mean_squared_error(result.fittedvalues, data[col]))

        log.info("SARIMAX Data Modelling Done")

        return mae, rmse

    def Prophet_modelling(self, data, params, future_num_points):

        '''
        Statistical Modelling
        this function is used to model data based on concepts of additive ordering
        :return: forecasted(predicted) data
        '''

        log.info("Initializing PROPHET Data Modelling")
        model = Prophet(growth=params['linear'],
                        n_changepoints=params['n_changepoints'],
                        changepoint_prior_scale=params['changepoint_prior_scale'],
                        changepoints=None,
                        holidays_prior_scale=params['holidays_prior_scale'],
                        interval_width=params['interval_width'],
                        holidays=None,
                        daily_seasonality=params['daily_seasonality'])

        model.fit(data)
        future = model.make_future_dataframe(future_num_points)
        forecast = model.predict(future)

        log.info("PROPHET Data Modelling Done")
        return forecast
