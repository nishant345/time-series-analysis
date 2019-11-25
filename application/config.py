from configparser import ConfigParser

config = ConfigParser()

config['setting'] = {
    'test': True
}

config['general'] = {
    'DATA_STORE_PATH': './Bike-Sharing-Dataset/hour.csv',
    'ML_LSTM_MODEL': False,
    'STATISTICAL_PROPHET_MODEL': False,
    'STATISTICAL_ARIMA_MODEL': True,
    'train_size': '0.7'
}

config['ARIMA'] = {
    'p': '0',
    'd': '0',
    'q': '2'
}


config['Prophet'] = {
    'model_name': 'Prophet',
    'cap': '0',
    'growth': 'linear',
    'n_changepoints': '25',
    'changepoints_prior_scale': '0.05',
    'changepoints' : '0',
    'holidays_prior_scale': '10',
    'interval_width': '0.8',
    'mcmc_samples': '0',
    'holidays': '0',
    'daily_seasonality': '1'
}

config['LSTM'] = {
    'model_name': 'LSTM',
    'lookback': '1',
    'loss': 'mean_squared_error',
    'optimizer': 'adam',
    'epochs': '3',
    'batch_size': '1',
    'verbose': '2',
}

with open('./config.ini', 'w') as f:
    config.write(f)