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


def MAE(y_true, y_pred):
    return Mean_Absolute_Error(y_true, y_pred)


def W1(y_true, forecasts, delay_distribution=None):
    """
    Computes the 1-Wasserstein distance between forecasted It trajectory convolved with the delay distribution and the observed case counts

    Args:
        y_true: Observed case counts
        forecasts: forecasted It trajectory
        delay_distribution: Delay distribution for symptom onset and case reporting

    Returns: The 1-Wasserstein distance
    """
    dtype = tf.float32

    delay_distribution = tf.squeeze(delay_distribution)
    delay_distribution = tf.cast(delay_distribution, dtype=dtype)
    delay_distribution = delay_distribution if delay_distribution.shape != tf.TensorShape(
        []) else tf.expand_dims(delay_distribution, axis=0)

    y_true = tf.squeeze(y_true)
    y_true = tf.cast(y_true, dtype=dtype)
    y_true = y_true if y_true.shape != tf.TensorShape(
        []) else tf.expand_dims(y_true, axis=0)

    forecasts = tf.squeeze(forecasts)
    forecasts = tf.cast(forecasts, dtype=dtype)
    forecasts = forecasts if forecasts.shape != tf.TensorShape(
        []) else tf.expand_dims(forecasts, axis=0)

    assert y_true.shape == forecasts.shape, "size of y_true and forecasts do not match"

    if delay_distribution is not None:
        l = len(delay_distribution)

        delay_distribution = tf.reshape(
            delay_distribution, shape=(l, 1, 1), name="Weights")
        init = tf.constant_initializer(delay_distribution.numpy())
        conv = Conv1D(1, l, kernel_initializer=init, use_bias=False)

        forecasts = tf.concat([forecasts, tf.zeros(l)], axis=0)
        forecasts = tf.reshape(forecasts, shape=(1, -1, 1))
        forecasts = conv(forecasts)
        forecasts = tf.squeeze(forecasts)

    x = range(y_true.shape[0])
    forecasts = forecasts[:y_true.shape[0]]
    d = wasserstein_distance(x, x, y_true, forecasts)

    return d
