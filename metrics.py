import numpy as np
import scipy.stats

def mse(prediction, target):
    return np.mean(np.square(prediction - target))


def mae(prediction, target):
    return np.mean(np.abs(prediction - target))


def rmse(prediction, target):
    return np.sqrt(mse(prediction, target))


def rmae(prediction, target):
    return np.sqrt(mae(prediction, target))


def entropy(y, proba = True):
    if y is None:
        return 0
    if isinstance(y, dict):
        classes = np.array(list(y.values()))
        if np.sum(classes) == 0:
            return 0
    else:
        if(len(y)) == 0:
            return 0
        classes = y.copy()
    if not proba:
        classes = classes / np.sum(classes)
    log_probs = 0 * classes
    log_probs[classes > 0] = np.log(classes[classes > 0])
    return -np.sum(classes * log_probs)
        

def gini(y, proba = True):
    if y is None:
        return 0
    if isinstance(y, dict):
        classes = np.array(list(y.values()))
        if np.sum(classes) == 0:
            return 0
    else:
        if(len(y)) == 0:
            return 0
        classes = y.copy()
    if not proba:
        classes = classes / np.sum(classes)

    classes = classes / np.sum(classes)
    return np.sum(classes * (np.ones(classes.shape) - classes))   


def mode(a):
    return scipy.stats.mode(a).mode


def probabilityDistribution(y, classes, laplaceSmoothing = 0):
    uniqueValues, count = np.unique(y, return_counts=True)
    probaDistribution = np.ones(len(classes)) * laplaceSmoothing
    for i, v in enumerate(uniqueValues):
        probaDistribution[classes[v]] += count[i]
    return probaDistribution / (len(y) + len(classes)*laplaceSmoothing)


def jaccard(prediction, target):
    pred = set(prediction)
    tar = set(target)
    return len(pred.intersection(tar))/len(pred.union(tar))


def accuracy(prediction, target, normalize = True, balanced = False):
    assert(prediction.shape == target.shape)
    if balanced:
        return (recall(prediction, target) + recall(prediction, target, 0))/2
    ans = np.sum(prediction == target)
    if normalize:
        ans /= len(prediction)
    return ans


# In this version, the follwoing classification metrics are only for binary classification
# The metrics work on numpy arrays and do not assert their type before checking


def precision(prediction, target, pos_label = 1):
    return np.sum(np.all([prediction == pos_label, target == pos_label], axis=0)) / np.sum(prediction == pos_label)    


def recall(prediction, target, pos_label = 1):
    return np.sum(np.all([prediction == pos_label, target == pos_label], axis=0)) / np.sum(target == pos_label)    


def f1(prediction, target, beta = 1):
    rec = recall(prediction, target)
    prec = precision(prediction, target)
    return (1+beta**2)*(prec*rec)/(beta**2 * prec + rec)

