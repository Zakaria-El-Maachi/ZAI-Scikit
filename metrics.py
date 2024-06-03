import numpy as np
import scipy.stats

def mse(prediction, target):
    """
    Calculate the Mean Squared Error (MSE) between predictions and targets.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.

    Returns:
    float: Mean Squared Error.
    """
    return np.mean(np.square(prediction - target))


def mae(prediction, target):
    """
    Calculate the Mean Absolute Error (MAE) between predictions and targets.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.

    Returns:
    float: Mean Absolute Error.
    """
    return np.mean(np.abs(prediction - target))


def rmse(prediction, target):
    """
    Calculate the Root Mean Squared Error (RMSE) between predictions and targets.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.

    Returns:
    float: Root Mean Squared Error.
    """
    return np.sqrt(mse(prediction, target))


def rmae(prediction, target):
    """
    Calculate the Root Mean Absolute Error (RMAE) between predictions and targets.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.

    Returns:
    float: Root Mean Absolute Error.
    """
    return np.sqrt(mae(prediction, target))


def entropy(y, proba=True):
    """
    Calculate the entropy of a distribution.

    Parameters:
    y (numpy.ndarray or dict): Target values or class counts.
    proba (bool): Whether the input is in probability format. Default is True.

    Returns:
    float: Entropy of the distribution.
    """
    if y is None:
        return 0
    if isinstance(y, dict):
        classes = np.array(list(y.values()))
        if np.sum(classes) == 0:
            return 0
    else:
        if len(y) == 0:
            return 0
        classes = y.copy()
    if not proba:
        classes = classes / np.sum(classes)
    log_probs = 0 * classes
    log_probs[classes > 0] = np.log(classes[classes > 0])
    return -np.sum(classes * log_probs)
        

def gini(y, proba=True):
    """
    Calculate the Gini impurity of a distribution.

    Parameters:
    y (numpy.ndarray or dict): Target values or class counts.
    proba (bool): Whether the input is in probability format. Default is True.

    Returns:
    float: Gini impurity of the distribution.
    """
    if y is None:
        return 0
    if isinstance(y, dict):
        classes = np.array(list(y.values()))
        if np.sum(classes) == 0:
            return 0
    else:
        if len(y) == 0:
            return 0
        classes = y.copy()
    if not proba:
        classes = classes / np.sum(classes)

    classes = classes / np.sum(classes)
    return np.sum(classes * (np.ones(classes.shape) - classes))   


def mode(a):
    """
    Calculate the mode of an array.

    Parameters:
    a (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Mode of the array.
    """
    return scipy.stats.mode(a).mode


def probability_distribution(y, classes, laplaceSmoothing=0):
    """
    Calculate the probability distribution of target values with optional Laplace smoothing.

    Parameters:
    y (numpy.ndarray): Target values.
    classes (dict): Dictionary of class labels to indices.
    laplaceSmoothing (float): Smoothing parameter. Default is 0.

    Returns:
    numpy.ndarray: Probability distribution of the target values.
    """
    uniqueValues, count = np.unique(y, return_counts=True)
    probaDistribution = np.ones(len(classes)) * laplaceSmoothing
    for i, v in enumerate(uniqueValues):
        probaDistribution[classes[v]] += count[i]
    return probaDistribution / (len(y) + len(classes) * laplaceSmoothing)


def jaccard(prediction, target):
    """
    Calculate the Jaccard similarity coefficient between predictions and targets.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.

    Returns:
    float: Jaccard similarity coefficient.
    """
    pred = set(prediction)
    tar = set(target)
    return len(pred.intersection(tar)) / len(pred.union(tar))


def accuracy(prediction, target, normalize=True, balanced=False):
    """
    Calculate the accuracy of predictions against targets.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.
    normalize (bool): Whether to normalize the result. Default is True.
    balanced (bool): Whether to use balanced accuracy. Default is False.

    Returns:
    float: Accuracy score.
    """
    assert prediction.shape == target.shape
    if balanced:
        return (recall(prediction, target) + recall(prediction, target, 0)) / 2
    ans = np.sum(prediction == target)
    if normalize:
        ans /= len(prediction)
    return ans

def accuracy_wrapper(estimator, X, y):
    """
    Wrapper function for calculating accuracy of an estimator.

    Parameters:
    estimator: Estimator object with a predict method.
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Target values.

    Returns:
    float: Accuracy score.
    """
    predictions = estimator.predict(X)
    return accuracy(predictions, y)


def precision(prediction, target, pos_label=1):
    """
    Calculate the precision score for binary classification.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.
    pos_label (int): The positive class label. Default is 1.

    Returns:
    float: Precision score.
    """
    return np.sum(np.all([prediction == pos_label, target == pos_label], axis=0)) / np.sum(prediction == pos_label)    


def recall(prediction, target, pos_label=1):
    """
    Calculate the recall score for binary classification.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.
    pos_label (int): The positive class label. Default is 1.

    Returns:
    float: Recall score.
    """
    return np.sum(np.all([prediction == pos_label, target == pos_label], axis=0)) / np.sum(target == pos_label)    


def f1(prediction, target, beta=1):
    """
    Calculate the F1 score for binary classification.

    Parameters:
    prediction (numpy.ndarray): Predicted values.
    target (numpy.ndarray): Actual values.
    beta (float): Weight of precision in the harmonic mean. Default is 1.

    Returns:
    float: F1 score.
    """
    rec = recall(prediction, target)
    prec = precision(prediction, target)
    return (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)
