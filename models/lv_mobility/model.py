import numpy
import scipy
from scipy.stats import gamma, norm


class lv_mobility:
    """
    An implementation of Larry and Valerie's mobility model 

    Attributes:
        args (dict): A dictionary containing values for A, alpha, beta, mu, and sig. Additional parameters can be included as well.
        fit: A method for fitting model parameters
        eval: A method for evaluating the model at timestep t
        batch_eval: A method for evaluation the model at timesteps between t1 and t2, inclusive.
    """

    def __init__(self, args={}):
        """Inits model parameters"""
        self.args = args

    def fit(self, M, DC, y_true, optimizer, loss, args):
        """
        Method for fitting model parameters

        Args:
            M (numpy array, shape=(num_features, num_intervals)): mobility time series
            DC (numpy array, shape=(num_intervals)): deatch curve
            y_true: observed values
            optimizer: optimizer for model
            loss: loss function
            args (dict): keys - ALPHA, BETA, MU : values - list of values for initialization
        """
        optimizer(self, M, DC, y_true, loss, args)

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
            1 and L
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

    def eval(self, M, DC, t, A, beta, alpha, mu, sig):
        """
        Method for evaluating the model at a specific timestep
        """
        assert(NotImplemented)

    def batch_eval(self, M, DC, t1, t2, A, beta, alpha, mu, sig):
        """
        Method for evaluating the model in a range of timesteps
        """
        assert(NotImplemented)
