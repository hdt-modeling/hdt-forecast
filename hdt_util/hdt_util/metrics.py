import numpy as np
import math
from scipy.stats import wasserstein_distance
import tensorflow as tf
from tensorflow.keras.layers import Conv1D


def Mean_Absolute_Error(y_true, y_pred):
    if isinstance(y_true, list):
        y_true = np.array(y_true).reshape(-1)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred).reshape(-1)
    assert y_true.shape[0] == y_pred.shape[0], '`y_true` and `y_pred` should have same length, but found {} vs {}'.format(
        y_true.shape[0], y_pred.shape[0])
    return np.mean(np.abs(y_true - y_pred))

def Wasserstein(y_true, forecasts, delay_dist):
    """
    Computes the 1-Wasserstein distance between the forecasted It trajectory
    convolved with the delay distribution and the observed case counts. We
    assume in our calcuation that the timesteps are integer valued. If you want
    to just compute the 1-Wasserstein distance between y_true and forecasts
    simply set delay_dest equal to [1].

    Args:
        y_true: Reported case counts
        forecasts: Forecasted values before convolving with the delay distribution
        delay_dist: Delay distribution for symptom onset and case reporting

    Returns: 
        The 1-Wasserstein distance
    """
    dtype = tf.float32

    delay_dist = tf.reshape(delay_dist, shape=-1)
    delay_dist = delay_dist[::-1]
    delay_dist = tf.cast(delay_dist, dtype=dtype)

    y_true = tf.reshape(y_true, shape=-1)
    y_true = tf.cast(y_true, dtype=dtype)

    forecasts = tf.reshape(forecasts, shape=-1)
    forecasts = tf.cast(forecasts, dtype=dtype)

    assert y_true.shape == forecasts.shape, "size of 'y_true' and 'forecasts' should have the same shape, but found {} vs {}".format(
        y_true.shape, forecasts.shape)

    l = len(delay_dist)

    delay_dist = tf.reshape(
        delay_dist, shape=(-1, 1, 1), name="Weights")
    init = tf.constant_initializer(delay_dist.numpy())
    conv = Conv1D(1, l, kernel_initializer=init, use_bias=False)

    forecasts = tf.concat([tf.zeros(l-1), forecasts, tf.zeros(l)], axis=0)
    forecasts = tf.reshape(forecasts, shape=(1, -1, 1))
    forecasts = conv(forecasts)
    forecasts = tf.reshape(forecasts, shape=-1)

    x = range(y_true.shape[0]+1)
    forecasts = forecasts[:y_true.shape[0]]
    
    #add an "extra day" in the end to balance the total number
    y_true = list(y_true.numpy())
    y_pred = list(forecasts.numpy())
    true_sum = sum(y_true)
    pred_sum = sum(y_pred)
    total = max(true_sum, pred_sum)
    y_true.append(total - true_sum)
    y_pred.append(total - pred_sum)
    d = wasserstein_distance(x, x, y_true, y_pred)

    return d


def MAE(y_true, y_pred):
    return Mean_Absolute_Error(y_true, y_pred)

def W1(y_true, y_pred, delay_dist):
    return Wasserstein(y_true, y_pred, delay_dist)
