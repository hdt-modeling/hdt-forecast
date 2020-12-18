"""
Adjust weekday effects via Poisson model.
"""
import cvxpy as cp
import numpy as np

from sklearn.model_selection import LeaveOneOut


class Weekday:
    """Class to handle weekday effects."""

    @staticmethod
    def get_params(sig, lam=10):
        """
        Estimate the fitted parameters of the Poisson model.
        Code taken from Aaron Rumack, with minor modifications.
        We model
           log(y_t) = alpha_{wd(t)} + phi_t
        where alpha is a vector of fixed effects for each weekday. For
        identifiability, we constrain \sum alpha_j = 0, and to enforce this we set
        Sunday's fixed effect to be the negative sum of the other weekdays.
        We estimate this as a penalized Poisson GLM problem with log link. We
        rewrite the problem as
            log(y_t) = X beta + log(denominator_t)
        and set a design matrix X with one row per time point. The first six columns
        of X are weekday indicators; the remaining columns are the identity matrix,
        so that each time point gets a unique phi. Hence, the first six entries of beta
        correspond to alpha, and the remaining entries to phi.
         The penalty is on the L1 norm of third differences of phi (so the third
        differences of the corresponding columns of beta), to enforce smoothness.
        Third differences ensure smoothness without removing peaks or valleys.
        Return a matrix of parameters: the entire vector of betas, for each time
        series column in the data.
        Args:
            sig: signal to adjust, array
            lam: penalty parameter, scalar
        Returns:
            beta: array of fitted parameters
        """
        # construct design matrix
        X = np.zeros((sig.shape[0], 6 + sig.shape[0]))
        not_sunday = np.where(sig.index.dayofweek != 6)[0]
        X[not_sunday, np.array(sig.index.dayofweek)[not_sunday]] = 1
        X[np.where(sig.index.dayofweek == 6)[0], :6] = -1
        X[:, 6:] = np.eye(X.shape[0])

        npsig = np.array(sig)
        beta = cp.Variable((X.shape[1]))
        lam_var = cp.Parameter(nonneg=True)
        lam_var.value = lam

        ll = ((cp.matmul(npsig, cp.matmul(X, beta)) -
               cp.sum(cp.exp(cp.matmul(X, beta)))
               ) / X.shape[0]
              )
        penalty = (lam_var * cp.norm(cp.diff(beta[6:], 3), 1) / (X.shape[0] - 2)
                   )  # L-1 Norm of third differences, rewards smoothness

        try:
            prob = cp.Problem(cp.Minimize(-ll + lam_var * penalty))
            _ = prob.solve()
        except:
            # If the magnitude of the objective function is too large, an error is
            # thrown; Rescale the objective function
            prob = cp.Problem(cp.Minimize((-ll + lam_var * penalty) / 1e5))
            _ = prob.solve()

        return beta.value

    @staticmethod
    def calc_adjustment(beta, y, dates):
        """
        Apply the weekday adjustment to a specific time series.
        Extracts the weekday fixed effects from the parameters and uses these to
        adjust the time series.
        Since
        log(y_t) = alpha_{wd(t)} + phi_t,
        we have that
        y_t = exp(alpha_{wd(t)}) exp(phi_t)
        and can divide by exp(alpha_{wd(t)}) to get a weekday-corrected ratio.
        """
        wd_correction = np.zeros((y.shape[0]))
        for wd in range(7):
            mask = dates == wd
            wd_correction[mask] = y[mask].value / (
                np.exp(beta[wd]) if wd < 6 else np.exp(-np.sum(beta[:6]))
            )
        return wd_correction


def dow_adjust_cases(loc_df, lam=None, lam_grid=None):
    """Wrapper func to do dow adjustment"""

    if lam_grid is None:
        lam_grid = [1, 10, 25, 75, 100]

    if lam is not None:
        params = Weekday.get_params(
            loc_df.groupby("time_value").sum().value, lam)
        return np.exp(params[6:])

    case_curve = loc_df.value.values
    N = case_curve.shape[0]
    loo = LeaveOneOut()
    lam_scores = []
    for lam in lam_grid:
        score = []
        for train_i, test_i in loo.split(case_curve):
            test_i = test_i[0]
            if 1 < test_i < (N - 2):
                raw_vals = loc_df.iloc[train_i].groupby(
                    "time_value").sum().value
                loo_params = Weekday.get_params(raw_vals, lam)
                fit = np.exp(loo_params[6:])
                preds_test = (fit[test_i - 1] + fit[test_i + 1]) / 2

                score.append(float(preds_test) - float(case_curve[test_i]))
        lam_scores.append(np.mean(np.square(score)))

    best_lam = lam_grid[np.argmin(lam_scores)]
    params = Weekday.get_params(loc_df.groupby(
        "time_value").sum().value, best_lam)

    return np.exp(params[6:])