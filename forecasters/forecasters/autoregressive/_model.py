import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from statsmodels.tsa.tsatools import lagmat
import numpy as np

class AR:
    """
    Simple autoregressive model with scaling correction.

    Attributes:
        p: Order of the autoregressive model, AR(p)
        beta: Parameters for the AR(p) model
        scale: Scaling parameter 
        lag: The time lag between the leading indicator and reported case counts
    """
    def __init__(self, p):
        """
        args:
            p (int): order of the AR model
        """
        self.p = p
        self.beta = None
        self.scale = 1.
        self.lag = 0

    def fit(self, x, lam=1):
        """
        Fit an AR(p) model on x.

        Args:
            x: observed 
            lam: regularization parameter
        """
        p = self.p
        n = x.shape[0]

        # lag matrix
        X = np.hstack((np.ones((n, 1)), lagmat(x, maxlag=p)))[p:]
        self.beta = np.linalg.solve(X.T @ X + lam*np.eye(p+1), X.T @ x[p:])

    def fit_scale(self, leading_indicator, cases, lag=0):
        """
        Scaling correction for AR(p) model to fit case counts.

        args:
            leading_indicator: Covariate that is a leading indicator for case counts
            cases (nd.array): 1-dimensional array containing case counts
            offset (int): Offset for window of cases counts you want to correct over
        """
        self.lag = lag

        leading_mean = np.mean(leading_indicator)
        cases_mean = np.mean(cases)
        leading_std = np.std(leading_indicator)
        cases_std = np.std(cases)

        z = leading_indicator[:-lag-1]
        x = cases[lag+1:]

        p = self.p
        n = z.shape[0]

        self.beta[0] += -leading_mean*cases_std/leading_std + cases_mean
        self.beta[1:] *= cases_std/leading_std

        Z = np.hstack([tf.ones((n, 1)), lagmat(z, maxlag=p)])[p:]
        x_pred = Z @ self.beta.T

        self.scale = np.sum(x)/np.sum(x_pred)

    def forecast(self, x, n=1):
        """
        It will only forecast based on the most recent entries in the time
        series supplied. For values beyond the n=1 step the method uses the
        previously forecasted values to make the next forecast.

        args:
            x (nd.array): array_like values to evaluate over
            n (int): How many time steps into the future you want to forecast
        """

        assert len(x) >= self.p, "Length of 'x' should at least be {} but found {}".format(
            self.p, len(x))
        p = self.p
        output = []

        for i in range(n):
            if i == 0:
                X1 = np.flip(x[-p:])
            else:
                X1 = np.concatenate([X1[1:], output[-1:]])

            X = np.concatenate([[1], X1])
            x_pred = self.beta.T @ X
            output.append(x_pred)

        return self.scale*np.array(output)


class ARLIC(tf.keras.Model):
    """
    Modified autoregressive model for fitting leading indicators to reported
    case counts. The models evaluates over the deconvolved leading indicator
    which is then reconvolved with the case report delay distribution. These
    values are then fitted to the actual case counts to obtain the model.

    Attributes:
        p: Order of the autoregressive model, AR(p)
        beta_conv: Convolutional layer for the AR(p) model parameters
        delay_dist: Case report delay distribution
        delay_conv: Convolutional layer for convolving with the delay distribution
        lag (int): The time lag between the leading indicator and reported case counts 
    """
    def __init__(self, p, delay_dist=[1]):
        super(ARLIC, self).__init__()
        """
        Args:
            p (int): order of the AR model
            delay_dist: array_like, delay distribution for symptom onset and
              case reporting
        """
        assert p>0 and isinstance(p, int), "p must be an integer greater than 0"

        self.p = p
        self.beta_conv = Conv1D(filters=1, kernel_size=p, use_bias=True)
        self.delay_dist = tf.reshape(
            delay_dist, shape=(-1, 1, 1), name="Weights")
        kernel_initializer = tf.constant_initializer(self.delay_dist.numpy())
        self.delay_conv = Conv1D(
            filters=1,
            kernel_size=self.delay_dist.shape[0],
            kernel_initializer=kernel_initializer,
            use_bias=False,
        )
        self.delay_conv.trainable = False
        self.lag = 0

    def call(self, x):
        return self.beta_conv(x)

    def forecast(self, x, n=1):
        """
        Method only forecasts based on the most recent entries in the time
        series supplied. For values beyond the n=1 step the method uses the
        previously forecasted values to make the next forecast.

        args:
            x (array_like): Values for the leading indicator
            n (int): How many time steps into the future you want to forecast
        """
        x = tf.reshape(x, shape=-1)
        assert len(x) >= self.p, "Length of 'x' should at least be {} but found {}".format(
            self.p, len(x))

        output = []
        x = tf.reshape(x, shape=(1, -1, 1))
        for _ in range(n):
            forecast = self(x)[:, -1:, :]
            output.append(forecast)
            x = tf.concat([x, forecast], axis=1)
        return tf.reshape(output, shape=-1)

    def train_step(self, inputs):
        leading_indicator, cases_reported = inputs
        leading_indicator = leading_indicator[:,:-1,:]
        cases_reported = cases_reported[:,self.p:,:]

        with tf.GradientTape() as tape:
            x_hat = self(leading_indicator, training=True)
            cases_forecasts = self.delay_conv(x_hat) 
            loss = self.loss(cases_reported, cases_forecasts)
        variables = [
            var for var in self.trainable_variables if var.name[:3] != "lag"]
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(
            zip(gradients, [var for var in self.trainable_variables if var.name[:3] != "lag"]))

        self.compiled_metrics.update_state(cases_reported, cases_forecasts)
        return {m.name: m.result() for m in self.metrics}
