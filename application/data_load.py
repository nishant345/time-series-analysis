import logging as log
import pandas as pd

# configuring log
log.basicConfig(filename='log/app.log', level=log.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')

class DataLoad():

    def __init__(self):
        log.info("DataLoad Object Created")
    
    def load_data(self, path):
        """
        get data from path
        :param path: URL where data is stored
        :return: data, Pandas data frame
        """
        try:
            data = pd.read_csv(path)
            log.info("Data loaded from data store path: {}".format(path))
            return data
        except:
            log.info("Invalid or no path given")
