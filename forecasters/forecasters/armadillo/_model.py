import numpy as np 
import scipy
from scipy.stats import gamma, norm
from scipy.optimize import dual_annealing

from ._loss import hellinger

DEFAULT_ARGS = {'ALPHA':0,
                'BETA':0,
                'MU':0,
                'bounds':np.array([
                                  [-100, 100],
                                  [-4, 4],
                                  [-40, 40],
                                  [-2, 2],
                                  [0.00001, 2],
                                 ]),
                'death_rate':0.03,
                'gamma_scale':3.64,
                'gamma_shape':6.28,
                'period':7,
                'prediction_upper_bound':5000}


class ArmadilloV1:
    """
    An implementation of Larry and Valerie's mobility model 

    Attributes:
        args (dict): A dictionary containing values for A, alpha, beta, mu, sig. It also will contain the M and DC used for latest fit.
        preds (numpy array): Best predictions stored from latest fit
        fit: A method for fitting model parameters
        eval: A method for evaluating the model at timestep t
        batch_eval: A method for evaluation the model at timesteps between t1 and t2, inclusive.
    """

    def __init__(self, args={}):
        """
        Inits model parameters
        
        Args:
            args (dict) : args for model training/tuning, default empty.
            Required parameters for the model includes 'ALPHA', 'BETA', 'MU', 'bounds', 'death_rate', 'gamma_scale', 'gamma_shape' and 'period'. If not provided in args, default values will be used.
        """
        
        self.args = args
        
        for key in DEFAULT_ARGS.keys():
            if args.get(key, None) is None:
                self.args[key] = DEFAULT_ARGS[key]

    def fit(self, args, loss=hellinger, initial_temp=20000, visit=2.0):
        """
        Method for fitting model parameters. The mobility time series M and death curve DC are also saved to args.
        Best predictions is saved to the preds variable.

        Args:
            args (dictionary) with keys:
                M (numpy array, shape=(num_intervals)): mobility time series
                DC (numpy array, shape=(num_intervals)): death curve
                y_true: observed values
            optimizer: optimizer for model
            loss: loss function
            initial_temp (float): initial temperature for dual annealing
            visit (float): another parameter for dual annealing
        """
        M = args['M']
        DC = args['DC']
        y_true = args['y_true']
        
        self.optim(M, DC, y_true, loss, initial_temp, visit)

        self.args["M"] = M
        self.args["DC"] = DC
        self.preds = self._eval(M, DC, len(
            M), self.args["A"], self.args["alpha"], self.args["beta"], self.args["mu"], self.args["sig"])
        
    def optim(self, M, DC, y_true, loss, initial_temp=20000, visit=2.0):
        """
        A method for constrained optimization using dual_annealing from scipy.optimize for lv_mobility.
    
        Args:
            M (numpy array): mobility time series
            DC (numpy array): death curve
            y_true (numpy array): observed value of cases
            loss: loss function 
            initial_temp (float): initial temperature for dual annealing
            visit (float): another parameter for dual annealing
        """
        
        def loss_wrap(x, *args):
            y_pred = self._eval(M, DC, L, x[0], x[1], x[2], x[3], x[4])
            return loss(y_true, y_pred)

        L = len(M)

        par_init = np.array([1.5+np.mean(y_true)/L, self.args['ALPHA'], self.args['BETA'], self.args['MU'], 1])

        args_ = [M, DC, L]
        res = dual_annealing(
            func=loss_wrap,
            args=args_,
            bounds=self.args['bounds'],
            x0=par_init,
            initial_temp=initial_temp,
            visit=visit,
            maxiter=200,
        )

        self.args['loss'] = res["fun"]
        self.args["A"] = res["x"][0]
        self.args["alpha"] = res["x"][1]
        self.args["beta"] = res["x"][2]
        self.args["mu"] = res["x"][3]
        self.args["sig"] = res["x"][4]

    def _eval(self, M, DC, L, A, alpha, beta, mu, sig):
        """
        Evaluation method for optimization

        Args:
            M (numpy array, shape=(num_intervals)): mobility time series
            DC (numpy array, shape=(num_intervals)): death curve 
            L (int): length of interval you want to evaluate over
            A (float): b_0 term in Larry and Valerie's doc
            alpha (float): multiplier for probit function
            beta (float): b_1 term in Larry and Valerie's doc
            mu (float): mean for normal distribution  in model
            sig (float): std for normal distribution in model

        Returns:
            A numpy array containing the predictions of the model between timesteps
            1 and L, inclusive
        """
        Beta = beta + alpha * norm.cdf(M[0:L], loc=abs(mu), scale=abs(sig))
        BetaSum = np.cumsum(np.insert(Beta, 0, 0))
        BetaSum = np.delete(BetaSum, L)

        # TODO(Stephen, Shitong): The initialization for the out variable needs to be tuned.
        # It fits poorly at beginning if A is large if all terms are nonzero
        # But it fits better at later timesteps with nonzero initialization
        # These simple initializations work decently but aren't optimal
        #out = (1/L) * np.ones(L)
        out = np.zeros(L)
        DC0 = DC
        for d in range(L):
            DC0 = np.insert(DC0, 0, 0)
            DC0 = np.delete(DC0, L)
            out = out + DC0*np.exp(BetaSum[d])
        out[np.isnan(out)] = self.args['prediction_upper_bound']
        out[out > self.args['prediction_upper_bound']] = self.args['prediction_upper_bound']
        out[out<0] = 0

        return abs(A)*out

    def forecast(self, l, M=[], DC=[], impute_method="same", impute_function=None):
        """
        Method for forecasting future values.

        Args:
            l (int): length of time interval to forecast
            M (numpy array, list): mobility time series
            DC (numpy array, list): death curve
            impute_method (string): same, custom, ... 
            impute_function (function): impute function for custom call

        """
        assert not (impute_method != "custom" and impute_function !=
                    None), "method incompatible with function"

        if not any(M):
            M = self.args["M"]

        # assumes mobility remains the same for future timesteps for the mobility time series
        # returns M with the values padded with the last value from M above for indexes up to l-1
        if impute_method == "same":
            L = len(M)
            M = np.concatenate([M, M[-1]*np.ones(l)])

        # TODO(Stephen, Shitong): Add other impute methods

        t = np.linspace(start=0, stop=l+L, num=l+L+1) #TODO: Vary t in a nonlinear way
        DC = gamma.pdf(t*self.args['period'], scale=self.args['gamma_scale'], a=self.args['gamma_shape'])  # a - shape parameter #TODO: Adaptively change scale and a as new data is ingested
        DC = (DC/np.sum(DC)) * self.args['death_rate']

        return self._eval(M, DC[:l+L], l+L, self.args["A"], self.args["alpha"], self.args["beta"], self.args["mu"], self.args["sig"])
