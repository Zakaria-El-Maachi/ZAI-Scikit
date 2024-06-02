from abc import ABC, abstractmethod
import warnings
import numpy as np
from joblib import Parallel, delayed
from metrics import accuracy
from sklearn.model_selection import cross_validate

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


class MetaEstimator(Estimator, ABC):
    def __init__(self, base_estimators=None):
        self.base_estimators = base_estimators if base_estimators else []

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class Pipeline(MetaEstimator):
    
    def __init__(self, steps):
        super().__init__(steps)

    def fit(self, X, y=None):
        for name, step in self.base_estimators[:-1]:
            X = step.fit_transform(X, y)
        self.base_estimators[-1][1].fit(X, y)
        return self

    def predict(self, X):
        for name, step in self.base_estimators[:-1]:
            X = step.transform(X)
        return self.base_estimators[-1][1].predict(X)

    def set_params(self, **params):
        step_params = {name: {} for name, step in self.base_estimators}
        for param, value in params.items():
            step_name, param_name = param.split('__', 1)
            step_params[step_name][param_name] = value
        for name, step in self.base_estimators:
            step.set_params(**step_params[name])
        return self

    def get_params(self, deep=True):
        if not deep:
            return super().get_params()
        out = {}
        for name, step in self.base_estimators:
            for param_name, value in step.get_params().items():
                out[f'{name}__{param_name}'] = value
        return out


class ModelSelector(MetaEstimator):

    def __init__(self, estimators, cv=5, scoring=None):
        super().__init__(estimators)
        self.cv = cv
        self.scoring = scoring
        self.best_score = None
        self.best_params = None
        self.best_estimator_ = None

    def fit(self, X, y):
        self.scores = []
        for name, estimator in self.base_estimators:
            scores = cross_validate(estimator, X, y, cv=self.cv, scoring=self.scoring)
            score = np.mean(scores['test_score'])
            self.scores.append((name, estimator, score))
        self.scores.sort(key=lambda x: x[2], reverse=True)
        best_name, self.best_estimator_, self.best_score = self.scores[0]
        self.best_params = self.best_estimator_.get_params()
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y):
        if self.scoring:
            return self.scoring(self.best_estimator_, X, y)
        else:
            # Default scoring: accuracy for classifiers
            return accuracy(self.predict(X), y)

    def get_scores(self):
        return [(name, score) for name, _, score in self.scores]


class DummyTransformer(Transformer):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X


class StackingClassifier(MetaEstimator, Classifier):
    def __init__(self, base_estimators, meta_estimator, scoring=None):
        super().__init__()
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.scoring = scoring 

    def fit(self, X, y):
        self.base_estimators_ = []
        for estimator in self.base_estimators:
            fitted_estimator = estimator.fit(X, y)
            self.base_estimators_.append(fitted_estimator)
        self.meta_features_ = self._get_meta_features(X)
        self.meta_estimator.fit(self.meta_features_, y)

    def _get_meta_features(self, X):
        meta_features = []
        for estimator in self.base_estimators_:
            predictions = estimator.predict(X)
            meta_features.append(predictions)
        return np.column_stack(meta_features)

    def predict(self, X):
        meta_features = self._get_meta_features(X)
        return self.meta_estimator.predict(meta_features)

    def score(self, X, y):
        return self.scoring(self.predict(X), y)
    
