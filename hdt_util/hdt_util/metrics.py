import numpy as np
import math

def Mean_Absolute_Error(y_true, y_pred):
    if isinstance(y_true, list):
        y_true = np.array(y_true).reshape(-1)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred).reshape(-1)
    assert y_true.shape[0] == y_pred.shape[0], '`y_true` and `y_pred` should have same length, but found {} vs {}'.format(y_true.shape[0], y_pred.shape[0])
    return np.mean(np.abs(y_true - y_pred))

def MAE(y_true, y_pred):
    return Mean_Absolute_Error(y_true, y_pred)