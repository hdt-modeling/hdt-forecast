import numpy
import scipy
from scipy.stats import gamma, norm
from lv_mobility.optimizer import optim
from lv_mobility.loss import hellinger


class LVMM:
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
        """Inits model parameters"""
        self.args = args
        self.preds = None

    def fit(self, M, DC, y_true, optimizer=optim, loss=hellinger, args=None):
        """
        Method for fitting model parameters. The mobility time series M and death curve DC are also saved to args.
        Best predictions is saved to the preds variable.

        Args:
            M (numpy array, shape=(num_intervals)): mobility time series
            DC (numpy array, shape=(num_intervals)): death curve
            y_true: observed values
            optimizer: optimizer for model
            loss: loss function
            args (dict): keys - ALPHA, BETA, MU : values - list of values for initialization
        """
        optimizer(self, M, DC, y_true, loss, args)

        self.args['M'] = M
        self.args['DC'] = DC
        self.preds = self._eval(M, DC, len(
            M), self.args['A'], self.args['alpha'], self.args['beta'], self.args['mu'], self.args['sig'])

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
        Beta = beta + alpha * \
            norm.cdf(M[0:L], loc=abs(mu), scale=abs(sig))
        BetaSum = numpy.cumsum(numpy.insert(Beta, 0, 0))
        BetaSum = numpy.delete(BetaSum, L)

        # TODO(Stephen, Shitong): The initialization for the out variable needs to be tuned.
        # It fits poorly at beginning if A is large if all terms are nonzero
        # But it fits better at later timesteps with nonzero initialization
        # These simple initializations work decently but aren't optimal
        #out = (1/L) * numpy.ones(L)
        out = numpy.zeros(L)
        DC0 = DC
        for d in range(L):
            DC0 = numpy.insert(DC0, 0, 0)
            DC0 = numpy.delete(DC0, L)
            out = out + DC0*numpy.exp(BetaSum[d])

        return abs(A)*out

    def eval(self, l1, l2):
        assert l1 < l2, "l1 should be less than l2"
        assert l2 <= len(self.args['predictions']
                        ) or l1 >= 0, "index is out of bounds"

        return self.args['predictions'][l1:l2]

    def forecast(self, l, M=[], DC=[], impute_method='same', impute_function=None):
        assert not (impute_method != "custom" and impute_function != None), "method incompatible with function"

        if not M:
            M = self.args["M"]

        if impute_method == 'same':
            L = len(M)
            M = numpy.concatenate([M, M[-1]*numpy.ones(l-L)])

        if not DC:
            t = numpy.linspace(start=0, stop=l, num=l+1)
            DC = gamma.pdf(t*7, scale=3.64, a=6.28)  # a - shape parameter
            DC = (DC/sum(DC)) * 0.03

        return self._eval(M, DC[:l], l, self.args['A'], self.args['alpha'], self.args['beta'], self.args['mu'], self.args['sig'])
