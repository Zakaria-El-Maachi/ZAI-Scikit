import numpy as np

class OneHotEncoder:
    """
    Encodes categorical integer features using a one-hot aka one-of-K scheme.
    """
    def __init__(self):
        self.uniques_ = {}

    def fit(self, X):
        """
        Fit OneHotEncoder to X.

        Parameters:
            X (array-like): Input array of shape (n_samples, n_features) to encode.
        """
        for idx in range(X.shape[1]):  # X.shape[1] is the number of features
            self.uniques_[idx] = np.unique(X[:, idx])
        return self

    def transform(self, X):
        """
        Transform X using one-hot encoding.

        Parameters:
            X (array-like): The input data to encode.

        Returns:
            array: The one-hot encoded representation of X.
        """
        output = np.zeros((X.shape[0], sum(len(u) for u in self.uniques_.values())))
        current_position = 0
        for idx, unique in self.uniques_.items():
            for i, val in enumerate(X[:, idx]):
                position = current_position + np.where(unique == val)[0][0]
                output[i, position] = 1
            current_position += len(unique)
        return output

    def fit_transform(self, X):
        """
        Fit and transform X with one-hot encoding.

        Parameters:
            X (array-like): The input data to encode.
        
        Returns:
            array: The one-hot encoded representation of X.
        """
        self.fit(X)
        return self.transform(X)


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    """
    def __init__(self):
        self.classes_ = {}

    def fit(self, X):
        """
        Initializes a new instance of the LabelEncoder.
        """
        self.classes_ = {label: idx for idx, label in enumerate(np.unique(X))}
        return self

    def transform(self, X):
        """
        Transform labels to normalized encoding.

        Parameters:
            X (array-like): The input data to encode.

        Returns:
            array: The encoded labels.
        """
        return np.array([self.classes_[item] for item in X])

    def fit_transform(self, X):
        """
        Fit and transform labels to normalized encoding.

        Parameters:
            X (array-like): The input data to encode.

        Returns:
            array: The encoded labels.
        """
        self.fit(X)
        return self.transform(X)


class OrdinalEncoder:
    """
    Encode categorical features using an ordinal encoding scheme.
    """
    def __init__(self, order):
        self.order = order
        self.mapping_ = {}

    def fit(self, X):
        for idx, value in enumerate(self.order):
            self.mapping_[value] = idx
        return self

    def transform(self, X):
        return np.array([self.mapping_[item] for item in X])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class BinaryEncoder:
    """
    Encode categorical features using an ordinal encoding scheme.
    """
    def __init__(self):
        self.unique_count_ = 0

    def fit(self, X):
        self.unique_count_ = len(np.unique(X))
        return self

    def transform(self, X):
        max_length = int(np.ceil(np.log2(self.unique_count_)))
        return np.array([[int(x) for x in np.binary_repr(val, width=max_length)] for val in X])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class FrequencyEncoder:
    """
    Encode categorical features by the frequency of each category.
    """
    def __init__(self):
        self.freq_map_ = {}

    def fit(self, X):
        unique, counts = np.unique(X, return_counts=True)
        total = float(len(X))
        self.freq_map_ = {key: val / total for key, val in zip(unique, counts)}
        return self

    def transform(self, X):
        return np.array([self.freq_map_[item] for item in X])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)