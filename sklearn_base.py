from abc import ABC, abstractmethod
import warnings
import numpy as np
from joblib import Parallel, delayed

class Estimator(ABC):
    
    @abstractmethod
    def fit(self, X, y = None):
        pass

    def set_params(self, **params):
        for parameter in params:
            if hasattr(self, parameter):
                setattr(self, parameter, params[parameter])
            else:
                warnings.warn(f"The {parameter} is not a valid parameter", UserWarning)

    def get_params(self, deep = True):
        params = self.__dict__
        if deep:
            return params.copy()
        return params
    

class Predictor(Estimator, ABC):
    
    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
    

class Classifier(Predictor, ABC):
    
    def __init__(self) -> None:
        self.classes = None

    
    @abstractmethod
    def predict_sample_proba(self, sample):
        pass

    def predict_proba(self, X):
        """
        Predicts the probability distribution for each input data point.

        Args:
            X (array-like): The input data.

        Returns:
            array-like: The predicted probability distributions.
        """
        return Parallel(n_jobs=-1)(delayed(self.predict_sample_proba)(sample) for sample in X)

    def predict(self, X):
        probaDistributions = self.predict_proba(X)
        classesList = sorted(self.classes.keys(), key = lambda x : self.classes[x])
        return Parallel(n_jobs=-1)(delayed(lambda x : classesList[np.argmax(x)])(probas) for probas in probaDistributions)
        

class Transformer(Estimator, ABC):
    @abstractmethod
    def transform(self, X, y = None):
        pass

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)
