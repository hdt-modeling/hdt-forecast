from forecasters.autoregressive._loss import mse
from forecasters import AR
import numpy as np
import tensorflow as tf


def test_AR_evaluation():
    model = AR(7)
    leading_indicator = np.random.uniform(100, 0, 1000)
    leading_indicator = tf.reshape(leading_indicator, shape=(1, -1, 1))
    forecasts = model(leading_indicator)
    forecasts = tf.reshape(forecasts, shape=-1)
    assert len(forecasts) == 994


def test_AR_forecast_method():
    model = AR(7)
    leading_indicator = np.random.uniform(100, 0, 1000)
    leading_indicator = tf.reshape(leading_indicator, shape=(1, -1, 1))

    forecasts = model.forecast(leading_indicator, n=1)
    forecasts = tf.reshape(forecasts, shape=-1)
    assert len(forecasts) == 1

    forecasts = model.forecast(leading_indicator, n=2)
    forecasts = tf.reshape(forecasts, shape=-1)
    assert len(forecasts) == 2


def test_AR_training():
    model = AR(7)
    optimizer = tf.keras.optimizers.Adam()
    loss = mse

    model.compile(
        optimizer=optimizer,
        loss=loss,
    )

    leading_indicator = np.random.uniform(100, 0, 1000)
    leading_indicator = tf.reshape(leading_indicator, shape=(1, -1, 1))

    reported_cases = np.random.uniform(100, 0, 1000)
    reported_cases = tf.reshape(leading_indicator, shape=(1, -1, 1))

    model.fit(
        x=leading_indicator,
        y=reported_cases,
        epochs=10000,
        verbose=0,
    )
