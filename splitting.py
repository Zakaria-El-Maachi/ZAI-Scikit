import numpy as np

def train_test_split(X, y, test_size=0.25, random_state=None):
    """
    Splits data into training and testing sets.
    
    Parameters:
        X (array-like): Input features dataset.
        y (array-like): Labels dataset.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): A seed value to ensure reproducibility.
    
    Returns:
        X_train, X_test, y_train, y_test: arrays representing the splits.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle arrays in unison
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split indices for training and testing
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test




class KFold:
    """
    K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).

    Parameters:
    n_splits (int): Number of folds. Must be at least 2.
    shuffle (bool): Whether to shuffle the data before splitting into batches.
    random_state (int): When shuffle is True, random_state affects the ordering of the indices. Pass an int for reproducible output across multiple function calls.

    Methods:
    get_n_splits(X, y, groups): Returns the number of splitting iterations in the cross-validator.
    split(X): Generates indices to split data into training and test set.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters:
        X (array-like): Always ignored, exists for compatibility.
        y (array-like): Always ignored, exists for compatibility.
        groups (array-like): Always ignored, exists for compatibility.

        Returns:
        int: Number of splits.
        """
        return self.n_splits

    def split(self, X):
        """
        Generate indices to split data into training and test set.

        Parameters:
        X (array-like): Training data, shape (n_samples, n_features).

        Yields:
        train_indices (array-like): The training set indices for that split.
        test_indices (array-like): The testing set indices for that split.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop
