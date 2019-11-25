import logging as log
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# configuring log
log.basicConfig(filename='log/app.log', level=log.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')

class Visualizer():
    def __init__(self):
        log.info("Visualizer Object Created")

    def plot_data_raw(self, data, title, x_label, y_label):

        '''
        plot raw time series data
        :param data: time series data
        :param title: title of the plot
        :param x_label: x label
        :param y_label: y label
        :return:
        '''

        #data.set_index('Timestamp', inplace=True)
        #data.index = pd.to_datetime(data.index)
        fig, ax = plt.subplots(1, 1)
        ax.plot(data)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def plot_rolling_sats(self, data, col=None):

        '''
        plot the rolling mean and average of data
        :param data: time series data
        :param col: column that needs to be averaged
        :return:
        '''
        rolmean = data[col].rolling(window=12, center=False).mean()
        rolstd = data[col].rolling(window=12, center=False).std()

        plt.plot(data[col], color='blue', label='original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling std')
        plt.legend(loc='best')
        plt.title("Rolling mean and Rolling std for %s" % (col))
        plt.xticks(rotation=45)
        plt.show()

    def plot_tranformed_data(self, data, col=None, col_transormed=None):

        '''
        plot the transformed data
        :param data: time series data
        :param col: original column
        :param col_transormed: transformed column
        :return:
        '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(data[col], color='blue')
        ax.plot(data[col_transormed], color='red')
        ax.set_title("%s and %s time series plot" % (col, col_transormed))
        ax.tick_params(axis="x", rotation=45)
        ax.legend([col, col_transormed])
        plt.show()

    def plot_acf_pacf(self, data, col=None, lag_acf=None, lag_pacf=None):

        '''
        plot the acf and pacf data
        :param data: transformed data
        :param col: original column
        :param lag_acf: acf data
        :param lag_pacf: pacf data
        :return:
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(lag_acf)
        ax1.axhline(y=0, linestyle='--', color="gray")
        ax1.axhline(y=-1.96 / np.sqrt(len(data[col])), linestyle='--', color="gray")
        ax1.axhline(y=1.96 / np.sqrt(len(data[col])), linestyle='--', color="gray")
        ax1.set_title("Autocorrelation Function for %s" % (col))

        ax2.plot(lag_pacf)
        ax2.axhline(y=0, linestyle='--', color="gray")
        ax2.axhline(y=-1.96 / np.sqrt(len(data[col])), linestyle='--', color="gray")
        ax2.axhline(y=1.96 / np.sqrt(len(data[col])), linestyle='--', color="gray")
        ax2.set_title("Partial Autocorrelation Function for %s" % (col))

        plt.tight_layout()
        plt.show()

    def plot_arima_model(self, data, result=None, col=None):

        '''
        :param data: original data
        :param result: predicted data
        :param col: target column
        :return:
        '''

        plt.plot(data[col])
        plt.plot(result.fittedvalues, color="red")
        plt.show()