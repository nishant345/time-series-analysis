import logging as log


# configuring log
log.basicConfig(filename='log/initial_data_analysis.log', level=log.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')

class InitialDataAnalysis():
    def __init__(self):
        log.info("InitialDataAnalysis Object Created")

    def _get_missing_values(self, data):
        """
        Find missing values of given data
        :param data: checked its missing value
        :return: Pandas Series object
        """

        # Getting missing values for each feature
        missing_values = data.isnull().sum()
        missing_values.sort_values(ascending=False, inplace=True)
        return missing_values

    def get_initial_info(self, data):
        """
        get feature name, data type, number of missing values and ten samples of each feature
        :param data: dataset information will be gathered from
        :return: no return value
        """
        feature_dtypes = data.dtypes
        self.missing_values = self._get_missing_values(data)

        print("=" * 50)

        print("{:16} {:16} {:25} {:16}".format("Feature Name".upper(),
                                               "Data Format".upper(),
                                               "# of Missing Values".upper(),
                                               "Samples".upper()))
        for feature_name, dtype, missing_value in zip(self.missing_values.index.values,
                                                      feature_dtypes[self.missing_values.index.values],
                                                      self.missing_values.values):
            print("{:18} {:19} {:19} ".format(feature_name, str(dtype), str(missing_value)), end="")
            for v in data[feature_name].values[:10]:
                print(v, end=",")
            print()

        print("=" * 50)

    def get_definition_data(self, data):
        """
        get description of the data set
        :param data: data set
        :return: no return value
        """
        print(data.describe())
        #log.info(a.count)



