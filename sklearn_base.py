from abc import ABC, abstractmethod
import warnings
import numpy as np

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
    def predictProba(self, X):
        pass

    def predict(self, X):
        probaDistributions = self.predictProba(X)
        classesList = sorted(self.classes.keys(), key = lambda x : self.classes[x])
        return np.vectorize(lambda x : classesList[np.argmax(x)], signature= '(n)->()')(probaDistributions)
        

class Transformer(Estimator, ABC):
    @abstractmethod
    def transform(self, X, y = None):
        pass

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)
