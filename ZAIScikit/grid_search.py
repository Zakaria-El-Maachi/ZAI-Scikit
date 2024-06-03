import numpy as np
from itertools import product

class GridSearchCV:
    """
    Exhaustive search over specified parameter values for an estimator.

    Parameters:
    estimator (object): The object of the model you want to use. This is assumed to implement the scikit-learn estimator interface.
    param_grid (dict): Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    scoring (callable): A scorer callable object / function with signature `scorer(estimator, X, y)`.
    cv (object): Cross-validation splitting strategy.

    Attributes:
    best_params_ (dict): Parameter setting that gave the best results on the hold out data.
    best_score_ (float): Score of the best estimator.
    results_ (list): List of all parameter settings along with their corresponding mean scores.
    """
    def __init__(self, estimator, param_grid, scoring, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.results_ = []

    def fit(self, X, y):
        """
        Run fit with all sets of parameters.

        Parameters:
        X (array-like): Training data.
        y (array-like): Target values.
        """
        param_names = list(self.param_grid.keys())
        param_combinations = list(product(*self.param_grid.values()))

        for params in param_combinations:
            # Update the model's parameters
            params_dict = dict(zip(param_names, params))
            self.estimator.set_params(**params_dict)

            scores = []
            for train_idx, test_idx in self.cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                self.estimator.fit(X_train, y_train)
                predictions = self.estimator.predict(X_test)
                score = self.scoring(y_test, predictions)
                scores.append(score)

            mean_score = np.mean(scores)
            self.results_.append((params_dict, mean_score))

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params_dict

    def get_results(self):
        """
        Get all parameter settings along with their corresponding mean scores.

        Returns:
        results (list): List of all parameter settings and their scores.
        """
        return self.results_

    def best_params(self):
        """
        Get the parameter setting that gave the best results on the hold out data.

        Returns:
        best_params (dict): Parameter setting that gave the best results.
        """
        return self.best_params_

    def best_score(self):
        """
        Get the score of the best estimator.

        Returns:
        best_score (float): Score of the best estimator.
        """
        return self.best_score_
