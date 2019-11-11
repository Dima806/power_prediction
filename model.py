import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor


# noinspection PyInterpreter
class PowerPrediction:
    '''
    Predict power consumption based on several estimates

    '''


    def __init__(self):
        '''

    Args:
        features_pkl_str (str)
        target_pkl_str (str)
        model (model) an instance of RandomForestRegressor
    Attributes:
        features (str): features dataframe
        target (str): target dataframe
        model: ML model
        '''
        print('>>> loading features')
        self.features = pd.read_pickle('features.pkl')
        print('>>> loading target')
        self.target = pd.read_pickle('target.pkl')
        print('>>> loading model')
        self.model = RandomForestRegressor(n_estimators=10,
                                           max_depth=4,
                                           min_samples_leaf=20,
                                           random_state=83)
        print(self.features.shape, self.target.shape, self.model)

    def preprocess(self, features, target):
        '''
        Preprocesses features and target dataframes

        :param features (dataframe): initial features dataframe
        :param target (dataframe): initial target dataframe
        :return: features (dataframe) - preprocessed features dataframe
        :return: target (dataframe) - preprocessed target dataframe
        '''

        for col in features.columns:
            # convert all values to float64
            features[col] = features[col].astype(np.float64)
            # convert all 0s to NaNs
            features.loc[features[col] == 0, col] = np.nan
        # create
        features['dt'] = features.index
        features.reset_index(inplace=True, drop=True)

        # add minute as feature
        features['minute'] = features['dt'].apply(lambda x: 60 * x.hour + x.minute)
        # add week of the year as feature
        features['week'] = features['dt'].apply(lambda x: x.isocalendar()[1])
        # add month as feature
        features['month'] = features['dt'].apply(lambda x: x.month)
        # add day of the year as feature
        features['day'] = features['dt'].apply(lambda x: 7 * (x.isocalendar()[1] - 1) +
                                                         x.isocalendar()[2] - 1)
        # remove dt column
        features.drop(['dt'], axis=1, inplace=True)
        # print(features.shape)
        target.reset_index(inplace=True, drop=True)
        return features, target

    def process_nans(self, features):
        '''
        Creates boolean isnull_ columns for each estimate column.
        Fills NaNs with zeros.

        :param features (dataframe) - preprocessed features with NaNs
        :return: features (dataframe) - features without NaNs
        '''
        for col in self.features.columns:
            features['isnull_{c}'.format(c=col)] = features[col].isnull()
        features.fillna(0, inplace=True)
        # print('shape', features.shape)
        # print(features.head())
        return features

    def fit(self, X, y):
        '''
        Fits model self.model with a given features and target

        :param X: features dataframe
        :param y: target dataframe
        :return: None
        '''
        self.model.fit(X, y)

    def predict(self, X):
        '''
        Predicts on features dataframe by using self.model.
        :param X: features dataframe for prediction
        :return: dataframe of predictions
        '''
        return self.model.predict(X)

    def score(self, X, y):
        '''
        Returns MAE score (less sensitive to outliers compared to RMSE)
        :param X: features dataframe
        :param y: target dataframe
        :return: score (float) - MAE score
        '''
        score = sklearn.metrics.mean_absolute_error(y, self.model.predict(X))
        return score
