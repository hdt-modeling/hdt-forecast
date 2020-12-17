from tensorflow.keras.losses import MSE
from forecasters.autoregressive import AR, ARLIC
import numpy as np
import tensorflow as tf


class TestARLIC:
    leading_indicator = np.random.uniform(100, 0, 1000)
    leading_indicator = tf.reshape(leading_indicator, shape=(1, -1, 1))

    reported_cases = np.random.uniform(100, 0, 1000)
    reported_cases = tf.reshape(leading_indicator, shape=(1, -1, 1))
    def test_evaluation(self):
        model = ARLIC(7)
        forecasts = model(self.leading_indicator)
        forecasts = tf.reshape(forecasts, shape=-1)
        assert len(forecasts) == 994

    def test_training(self):
        model = ARLIC(7)
        optimizer = tf.keras.optimizers.Adam()
        loss = MSE

        model.compile(
            optimizer=optimizer,
            loss=loss,
        )

        args = {
            "x": self.leading_indicator,
            "y": self.reported_cases,
            "epochs": 100,
            "verbose": 1,
            "callbacks": None,
        }
        model.fit(args)

    def test_forecast_method(self):
        model = ARLIC(7)
        forecasts = model.forecast(self.leading_indicator, n=1)
        forecasts = tf.reshape(forecasts, shape=-1)
        assert len(forecasts) == 1

        forecasts = model.forecast(self.leading_indicator, n=2)
        forecasts = tf.reshape(forecasts, shape=-1)
        assert len(forecasts) == 2

