import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true-y_pred) ** 2)


def l1(y_true, y_pred):
    return np.sum(abs(y_true-y_pred))


def hellinger(y_true, y_pred):
    return np.sum((y_true**0.5-y_pred**0.5)**2)
