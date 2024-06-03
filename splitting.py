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
