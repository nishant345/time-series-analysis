'''
There are many methods that can be added here.
But add only those methods which you are going to use.
YAGNI principle
'''

import logging as log
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# configuring log
log.basicConfig(filename='log/app.log', level=log.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')


class PreProcessing():
    def __init__(self):
        log.info("PreProcessing Object Created")


    def remove_columns(self):
        log.info(" testing column removed function ")

    '''
    *****-------------------- PRE PROCESSING DATA FOR LSTM ---------------------------******
    
    '''
    def data_normailstaion(self, data):

        '''
        normalising the data set
        :param data: time series data
        :return: normalised data set
        '''

        dataset = data
        dataset = dataset.reshape(len(dataset), 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        log.info(" Data Normalisation Done ")
        return dataset


    def _create_dataset_for_LSTM(self, dataset, look_back):
        '''
        convert an array of values into a data set matrix
        :param dataset: array of values
        :param look_back: look back for data set
        :return: data set matrix
        '''
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def reshape_datset_for_LSTM(self, data_train, data_test, lookback):

        '''
        reshape into X=t and Y=t+1
        :return:
        '''
        look_back = lookback
        trainX, trainY = self._create_dataset_for_LSTM(data_train, look_back)
        testX, testY = self._create_dataset_for_LSTM(data_test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        print(trainX.shape, trainY.shape, testX.shape, testY.shape)
        return trainX, trainY, testX, testY

    def data_denormailstaion(self, trainPredict, testPredict, trainY, testY):

        '''
        denormalising the data set
        :param data: predicted  data for train and test
        :return: denormalised data set
        '''

        scaler = MinMaxScaler(feature_range=(0, 1))
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform(testY.reshape(-1, 1))
        return trainPredict, testPredict, trainY, testY


