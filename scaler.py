import numpy as np
from sklearn_base import Transformer

class MinMaxScaler(Transformer):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X, y=None):
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        feature_range_min, feature_range_max = self.feature_range
        self.scale_ = (feature_range_max - feature_range_min) / self.data_range_
        self.min_ = feature_range_min - self.data_min_ * self.scale_
        return self

    def transform(self, X, y=None):
        return self.scale_ * X + self.min_

    def inverse_transform(self, X, y=None):
        return (X - self.min_) / self.scale_
    

class StandardScaler(Transformer):
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        self.scale_ = np.sqrt(self.var_)
        return self

    def transform(self, X, y=None):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X, y=None):
        return X * self.scale_ + self.mean_