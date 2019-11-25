import logging as log
import pandas as pd
from datetime import datetime

# configuring log
log.basicConfig(filename='log/app.log', level=log.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')

class Utility():
    def __init__(self):
       log.info("Utility Object Created")

    def days_between(self, d1, d2):

        '''
        counting the days between two dates
        :param d1: first date
        :param d2: second date
        :return: count of days
        '''
        d1 = datetime.strptime(d1, '%Y-%m-%d')
        d2 = datetime.strptime(d2, '%Y-%m-%d')
        return abs((d2 - d1).days + 1)

    def add_col_timestamp(self, df):

        '''
        adding the timestamp column
        :param df: original time series data
        :return: df with timestamp
        '''

        df['hr'] = df['hr'].astype(str)
        df['Timestamp'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'] + ':00' + ':00', format='%Y-%m-%d %H:%M:%S')
        log.info("Utility Object Created")
        return df

    def change_to_float32(self, data):

        '''
        change the dataset to float array
        :param data: data set
        :return: float data set
        '''

        dataset = data
        # converting to numpy array
        dataset = dataset.values
        dataset = dataset.astype('float32')
        log.info("data converted to float32 array")
        return dataset

    def split_train_test_data(self, data, k):

        '''
        split the data set into train and test
        :param data: data set
        :param k: percent of splitting
        :return: train and test data set
        '''

        dataset = data
        train_size = int(k * len(data))
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        print(len(train), len(test))
        return train, test