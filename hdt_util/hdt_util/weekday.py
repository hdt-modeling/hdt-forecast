"""
Adjust weekday effects via Poisson model.
"""
import cvxpy as cp
import numpy as np

from sklearn.model_selection import LeaveOneOut


class Weekday:
    """Class to handle weekday effects."""

    @staticmethod
    def get_params(counts, dayofweek, lam=10):
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

        Params:
        =======
            counts: List<float>, counts data to be adjusted
            dayofweek: List<int>, day of the week for each count
            lam: float, optional, default 10, penalty parameter

        Returns:
        ========
            beta: array<float>, fitted parameters
        """
        
        # construct design matrix
        L = len(counts)
        X = np.zeros((L, 6+L))
        for ind, day in enumerate(dayofweek):
            if day!=6:
                X[ind, day] = 1
            else:
                X[ind, :6] = -1
            X[ind, 6+ind] = 1

        counts = np.array(counts)
        beta = cp.Variable((6+L))
        lam_var = cp.Parameter(nonneg=True)
        lam_var.value = lam

        ll = cp.matmul(counts, cp.matmul(X, beta)) - cp.sum(cp.exp(cp.matmul(X, beta))) 
        ll = ll / L
        diff = cp.diff(beta[6:], 3) # Here it has to be assumed that the data is continuous
        penalty = lam_var * cp.norm(diff, 1) / (L - 2)# L-1 Norm of third differences, rewards smoothness

        try:
            #Why don't directly use the second?
            prob = cp.Problem(cp.Minimize(-ll + lam_var * penalty))
            _ = prob.solve()
        except:
            # If the magnitude of the objective function is too large, an error is
            # thrown; Rescale the objective function
            prob = cp.Problem(cp.Minimize((-ll + lam_var * penalty) / 1e5))
            _ = prob.solve()

        return beta.value

    @staticmethod
    def calc_adjustment(beta, counts, dayofweek):
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
        
        counts = np.array(counts).reahpe(-1)
        dayofweek = np.array(dayofweek)
        beta = beta[:7]
        beta[6] = -sum(beta[:6])
        beta = np.exp(beta)
        
        correction = np.zeros(counts.shape)
        for day in range(7):
            mask = dayofweek == day
            correction[mask] = counts[mask] / beta[day]
        return correction


def dow_adjust_cases(counts, dayofweek, lam=None, lam_grid=[1, 10, 25, 75, 100]):
    """
    Apply day of week adjustment with given choice of lambda, or choose one with LOO loss
    
    Params:
    =======
        counts: List<float>, counts data to be adjusted
        dayofweek: List<int>, day of the week for each count
        lam: float, optional, default None, penalty parameter if None, the best one in lam_grid will be used
        lam_grid: List<float>, possible values for lam for tuning purpose
    """

    if lam is not None:
        params = Weekday.get_params(counts, dayofweek, lam)
        return np.exp(params[6:])
    
    N = len(counts)
    lam_scores = []
    for lam in lam_grid:
        errors = []
        for test_ind in range(N):
            # One problem here is that, Weekday.get_params assumes that the data is continuous. 
            # But when we are doing this Leave-one-out corss validation, we remove one observation directly
            # To compensate for this, we do not directly remove the value,
            # but use the average value of one day before and one day after
            # Originally in Maria's code this problem is ignored and the loss calculation is also wrong.
            # She used the average of one day before and two days after as prediction
            # TODO:
            #     Find a suitable way to conduct this LOO parameter tuning
            train_counts = counts.copy()
            if test_ind == 0:
                left = counts[0]
                right = counts[1]
            elif test_ind == N-1:
                left = counts[-2]
                right = counts[-1]
            else:
                left = counts[test_ind-1]
                right = counts[test_ind+1]
            train_counts[test_ind] = (train_counts[left] + train_counts[right]) / 2
            loo_params = Weekday.get_params(train_counts, dayofweek, lam)
            fit = np.exp(loo_params[6:])
            errors.append(fit[test_ind] - counts[test_ind])
            
        lam_scores.append(np.mean(np.square(errors)))

    best_lam = lam_grid[np.argmin(lam_scores)]
    params = Weekday.get_params(counts, dayofweek, best_lam)

    return np.exp(params[6:])