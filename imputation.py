import numpy as np
from linear_regression import LinearRegression
from scipy.stats import mode

class SimpleImputer:
    """
    Imputation transformer for completing missing values.

    Parameters:
    strategy (str): The imputation strategy.
                    - "mean": Replace missing values using the mean along each column.
                    - "median": Replace missing values using the median along each column.
                    - "mode": Replace missing values using the most frequent value along each column.
                    - "constant": Replace missing values with fill_value.
    fill_value (any): When strategy == "constant", fill_value is used to replace all occurrences of missing values.

    Attributes:
    statistics_ (list): The statistics (mean, median, or mode) for each column in the training set.
    """
    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        """
        Compute the statistics for each column.

        Parameters:
        X (array-like): The input data with missing values.

        Returns:
        self: Returns self.
        """
        if self.strategy in ["mean", "median", "mode"]:
            self.statistics_ = []
            for col in range(X.shape[1]):
                if self.strategy == "mean":
                    self.statistics_.append(np.nanmean(X[:, col]))
                elif self.strategy == "median":
                    self.statistics_.append(np.nanmedian(X[:, col]))
                elif self.strategy == "mode":
                    self.statistics_.append(mode(X[:, col], nan_policy='omit').mode[0])
        return self

    def transform(self, X):
        """
        Replace missing values using the fitted statistics.

        Parameters:
        X (array-like): The input data with missing values.

        Returns:
        X_transformed (array-like): The input data with missing values imputed.
        """
        X_transformed = np.array(X, copy=True)
        if self.strategy == "constant":
            fill_values = self.fill_value
        else:
            fill_values = self.statistics_

        for i, fv in enumerate(fill_values):
            if fv is not None:
                X_transformed[np.isnan(X_transformed[:, i]), i] = fv
        return X_transformed

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters:
        X (array-like): The input data with missing values.

        Returns:
        X_transformed (array-like): The input data with missing values imputed.
        """
        return self.fit(X).transform(X)


class KNNImputer:
    """
    Imputation for completing missing values using k-Nearest Neighbors.

    Parameters:
    n_neighbors (int): Number of neighboring samples to use for imputation.

    Attributes:
    n_neighbors (int): Number of neighboring samples to use for imputation.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def _compute_distances(self, X_complete, X_missing):
        """
        Compute the distances between missing and non-missing samples.

        Parameters:
        X_complete (array-like): The input data without missing values.
        X_missing (array-like): The input data with missing values.

        Returns:
        distances (array-like): The distances between each missing sample and each non-missing sample.
        """
        n_missing = X_missing.shape[0]
        n_complete = X_complete.shape[0]
        distances = np.empty((n_missing, n_complete))

        for i in range(n_missing):
            for j in range(n_complete):
                diff = X_complete[j, :] - X_missing[i, :]
                sq_diff = diff ** 2
                valid_diff = ~np.isnan(sq_diff)
                distances[i, j] = np.sqrt(np.sum(sq_diff[valid_diff]))
        
        return distances

    def fit_transform(self, X):
        """
        Fit to data, then transform it using k-Nearest Neighbors imputation.

        Parameters:
        X (array-like): The input data with missing values.

        Returns:
        X_filled (array-like): The input data with missing values imputed.
        """
        X_filled = np.copy(X)
        n_samples, n_features = X.shape

        for feature in range(n_features):
            missing_mask = np.isnan(X[:, feature])
            non_missing_mask = ~missing_mask
            
            if np.any(missing_mask):
                X_missing = X[missing_mask]
                X_complete = X[non_missing_mask]
                
                distances = self._compute_distances(X_complete[:, [feature]], X_missing[:, [feature]])
                nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
                
                for i, missing_index in enumerate(np.where(missing_mask)[0]):
                    nearest_values = X[non_missing_mask][nearest_neighbor_indices[i], feature]
                    X_filled[missing_index, feature] = np.nanmean(nearest_values)

        return X_filled


class IterativeImputer:
    """
    Imputation for completing missing values using iterative model-based imputation.

    Parameters:
    max_iter (int): Maximum number of imputation iterations.
    n_iter (int): Number of iterations for the Linear Regression model.
    alpha (float): Learning rate for the Linear Regression model.

    Attributes:
    models (dict): Dictionary to store trained Linear Regression models for each feature.
    """
    def __init__(self, max_iter=10, n_iter=5000, alpha=0.01):
        self.max_iter = max_iter
        self.n_iter = n_iter
        self.alpha = alpha
        self.models = {}

    def fit_transform(self, X):
        """
        Fit to data, then transform it using iterative model-based imputation.

        Parameters:
        X (array-like): The input data with missing values.

        Returns:
        X_filled (array-like): The input data with missing values imputed.
        """
        X_filled = np.array(X, dtype=float)  # Ensure working with a float copy for NaN handling
        n_samples, n_features = X.shape

        for _ in range(self.max_iter):
            for feature in range(n_features):
                target = X_filled[:, feature]
                is_missing = np.isnan(target)
                if not np.any(is_missing):
                    continue  # Skip if no missing values

                # Features for training: use all features except the target feature
                X_train = np.delete(X_filled, feature, axis=1)
                y_train = target[~is_missing]
                X_train = X_train[~is_missing, :]
                
                # Train the model
                model = LinearRegression(n_iter=self.n_iter, alpha=self.alpha)
                model.fit(X_train, y_train)
                
                # Store model for possible later use
                self.models[feature] = model
                
                # Predict and fill missing values
                X_test = X_filled[is_missing, :]
                X_test = np.delete(X_test, feature, axis=1)
                X_filled[is_missing, feature] = model.predict(X_test)

        return X_filled
