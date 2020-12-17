from tensorflow.keras.losses import MSE
from forecasters.autoregressive import AR, ARLIC
import numpy as np
import tensorflow as tf


def test_ARLIC_evaluation():
    model = ARLIC(7)
    leading_indicator = np.random.uniform(100, 0, 1000)
    leading_indicator = tf.reshape(leading_indicator, shape=(1, -1, 1))
    forecasts = model(leading_indicator)
    forecasts = tf.reshape(forecasts, shape=-1)
    assert len(forecasts) == 994


def test_ARLIC_forecast_method():
    model = ARLIC(7)
    leading_indicator = tf.random.uniform((1,100,1),0,1000)

    forecasts = model.forecast(leading_indicator, n=1)
    forecasts = tf.reshape(forecasts, shape=-1)
    assert len(forecasts) == 1

    forecasts = model.forecast(leading_indicator, n=2)
    forecasts = tf.reshape(forecasts, shape=-1)
    assert len(forecasts) == 2


def test_ARLIC_training():
    model = ARLIC(7)
    optimizer = tf.keras.optimizers.Adam()
    loss = MSE

    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    leading_indicator = np.random.uniform(100, 0, 1000)
    leading_indicator = tf.reshape(leading_indicator, shape=(1, -1, 1))

    reported_cases = np.random.uniform(100, 0, 1000)
    reported_cases = tf.reshape(leading_indicator, shape=(1, -1, 1))

    args = {
        "x": leading_indicator,
        "y": reported_cases,
        "epochs": 100,
        "verbose": 1,
        "callbacks": None,
    }
    model.fit(args)
