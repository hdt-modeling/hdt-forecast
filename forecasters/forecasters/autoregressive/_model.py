import tensorflow as tf
from statsmodels.tsa.tsatools import lagmat
import numpy as np


class AR:
    def __init__(self, p):
        """
        AR(p) model with scaling correction.

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
            lam:
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
        Forecasting trajectories for cases counts

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
