# Standard modules
import pandas as pd
import logging as log
from configparser import ConfigParser

import warnings
warnings.filterwarnings("ignore")


# Custom modules
from application.data_load import DataLoad
from application.exploratory_data_analysis import ExploratoryDataAnalysis
from application.initial_data_analysis import InitialDataAnalysis
from application.pre_processing import PreProcessing
from application.data_modelling import DataModelling
from application.visualizer import Visualizer
from application.utility import Utility



# configuring log
log.basicConfig(filename='log/app.log', level=log.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')

def main():

    # Instantiating classes
    dl = DataLoad()
    eda = ExploratoryDataAnalysis()
    ida = InitialDataAnalysis()
    pp = PreProcessing()
    dm = DataModelling()
    visual = Visualizer()
    util = Utility()

    #congig parser
    parser = ConfigParser()
    parser.read('config.ini')

    # data store path
    DATA_STORE_PATH = parser.get('general', 'data_store_path')

    # general settings; which model to run
    ML_LSTM_MODEL = eval(parser.get('general', 'ml_lstm_model'))
    STATISTICAL_PROPHET_MODEL = eval(parser.get('general', 'statistical_prophet_model'))
    STATISTICAL_ARIMA_MODEL = eval(parser.get('general', 'statistical_arima_model'))

    # Load data set
    data_df = dl.load_data(DATA_STORE_PATH)

    # adding timestamp column to data frame
    data_df = util.add_col_timestamp(data_df)

    # getting the initial statistics about data
    ida.get_initial_info(data_df)

    # gat the description of target column
    ida.get_definition_data(data_df["cnt"])

    # visualize the original time series
    visual.plot_data_raw(data=data_df["cnt"], title="Timestamp Vs cnt graph", x_label="Timestamp", y_label="cnt")

    if STATISTICAL_ARIMA_MODEL:
        '''
            ********---------------------------------------------------------------------------------------------------********

                                             STATISTICAL MODELLING USING ARIMA 

           ********---------------------------------------------------------------------------------------------------********
        '''

        df_arima = pd.DataFrame({'ds': data_df['Timestamp'], 'y': data_df['cnt']})

        # Plotting original time series
        visual.plot_data_raw(df_arima, "Time Series data", 'ds', 'cnt')

        # Plotting rolling mean and rolling std to check for Stationarity
        visual.plot_rolling_sats(df_arima, col='y')

        # Test for stationarity Dickey-Fuller Test
        eda.test_stationarity(df_arima, col='y')

        # Transformation of data in log, moving average, log_diff etc.
        transformed_data = eda.data_transformation(df_arima)

        # visualize transformed data
        visual.plot_tranformed_data(transformed_data, col='y', col_transormed='y_log')
        visual.plot_tranformed_data(transformed_data, col='y', col_transormed='y_log_moving_avg')
        visual.plot_tranformed_data(transformed_data, col='y', col_transormed='y_moving_avg')
        visual.plot_tranformed_data(transformed_data, col='y', col_transormed='y_log_diff')
        visual.plot_tranformed_data(transformed_data, col='y', col_transormed='y_log_ewma')
        visual.plot_tranformed_data(transformed_data, col='y', col_transormed='y_log_ewma_diff')

        # Plotting rolling mean and rolling std to check for transformed data
        visual.plot_rolling_sats(transformed_data, col='y_log')
        visual.plot_rolling_sats(transformed_data, col='y_log_moving_avg')
        visual.plot_rolling_sats(transformed_data, col='y_moving_avg')
        visual.plot_rolling_sats(transformed_data, col='y_log_diff')
        visual.plot_rolling_sats(transformed_data, col='y_log_ewma')
        visual.plot_rolling_sats(transformed_data, col='y_log_ewma_diff')

        # Test for stationarity Dickey-Fuller Test on transformed data
        eda.test_stationarity(transformed_data, col='y_log')
        eda.test_stationarity(transformed_data, col='y_log_moving_avg')
        eda.test_stationarity(transformed_data, col='y_moving_avg')
        eda.test_stationarity(transformed_data, col='y_log_diff')
        eda.test_stationarity(transformed_data, col='y_log_ewma')
        eda.test_stationarity(transformed_data, col='y_log_ewma_diff')

        # find the acf and pacf values of the transformed data
        lag_acf, lag_pacf  = eda.get_acf_pacf(transformed_data)

        # plot ACF and PACF plots
        visual.plot_acf_pacf(transformed_data, col="y_log_diff", lag_acf=lag_acf, lag_pacf=lag_pacf)

        # parameters for ARIMA model
        p = int(parser.get('ARIMA', 'p'))
        d = int(parser.get('ARIMA', 'd'))
        q = int(parser.get('ARIMA', 'q'))

        # ARIMA train and predict; error metric = MAE
        mae, rmse, result = dm.ARIMA_modelling(transformed_data, col='y_log_diff', p=p, d=d, q=q)
        log.info("for ARIMA model %i %i %i for %s, MAE: %.4f RMSE: %.4f" % (p, d, q, 'y_log_diff', mae, rmse))
        print("for ARIMA model %i %i %i for %s, MAE: %.4f RMSE: %.4f" % (p, d, q, 'y_log_diff', mae, rmse))

        # visual.plot_arima_model(transformed_data, result, col="y_log_diff")

    if STATISTICAL_PROPHET_MODEL:
        '''
            ********---------------------------------------------------------------------------------------------------********

                                             STATISTICAL MODELLING USING PROPHET 

           ********---------------------------------------------------------------------------------------------------********
        '''
        df_prophet = pd.DataFrame({'ds': data_df['Timestamp'], 'y': data_df['cnt']})

        start_training_date = parser.get('Prophet', 'start_training_date')
        end_training_date = parser.get('Prophet', 'end_training_date')


        # configuring parameters for Prophet model
        future_num_points = util.days_between(end_training_date, start_training_date)
        params_prophet = dict()
        params_prophet['cap'] = int(parser.get('Prophet', 'cap'))
        params_prophet['growth'] = parser.get('Prophet', 'growth')
        params_prophet['n_changepoints'] = int(parser.get('Prophet', 'n_changepoints'))
        params_prophet['changepoints_prior_scale'] = float(parser.get('Prophet', 'changepoints_prior_scale'))
        params_prophet['changepoints'] = int(parser.get('Prophet', 'changepoints'))
        params_prophet['holidays_prior_scale'] = int(parser.get('Prophet', 'holidays_prior_scale'))
        params_prophet['interval_width'] = float(parser.get('Prophet', 'interval_width'))
        params_prophet['holidays'] = int(parser.get('Prophet', 'holidays'))
        params_prophet['daily_seasonality'] = int(parser.get('Prophet', 'daily_seasonality'))

        forecast = dm.Prophet_modelling(df_prophet, params_prophet, future_num_points)



    if ML_LSTM_MODEL:
        '''
            ********---------------------------------------------------------------------------------------------------********
            
                                            MACHINE LEARNING MODELLING USING LSTM
            
            ********---------------------------------------------------------------------------------------------------********
    
        '''
        # configuring parameters for LSTM model
        train_size = float(parser.get('general', 'train_size'))
        lookback = int(parser.get('LSTM', 'lookback'))
        params_LSTM = dict()
        params_LSTM["loss"] = parser.get('LSTM', 'loss')
        params_LSTM["optimizer"] = parser.get('LSTM', 'optimizer')
        params_LSTM["epochs"] = int(parser.get('LSTM', 'epochs'))
        params_LSTM["batch_size"] = int(parser.get('LSTM', 'batch_size'))
        params_LSTM["verbose"] = int(parser.get('LSTM', 'verbose'))
        params_LSTM["lookback"] = int(parser.get('LSTM', 'lookback'))


        # converting the data set cnt values to float
        data_df = util.change_to_float32(data_df["cnt"])

        # normalising the data
        data_df = pp.data_normailstaion(data_df)

        # split into train and test sets
        # 70 % train_set 30% test_set
        train_set, test_set = util.split_train_test_data(data_df, train_size)

        # reshape input to be [samples, time steps, features]
        trainX, trainY, testX, testY = pp.reshape_datset_for_LSTM(train_set, test_set, lookback=lookback)

        # create and fit the LSTM network
        train_predict, test_predict = dm.LSTM_Modelling(trainX, trainY, testX, testY, params_LSTM)

        # de-normalising the data
        trainPredict, testPredict, trainY, testY = pp.data_denormailstaion(train_predict, test_predict, trainY, testY)
    


if __name__ == "__main__":
    main()