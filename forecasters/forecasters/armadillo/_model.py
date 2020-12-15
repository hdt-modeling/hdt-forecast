import numpy as np
import scipy
from scipy.stats import gamma, norm
from scipy.optimize import dual_annealing
import tensorflow as tf
from ._loss import hellinger
from ._layers import LinearLayer
tf.keras.backend.set_floatx('float64')
dtype = tf.float64

DEFAULT_ARGS = {'ALPHA': 0,
                'BETA': 0,
                'MU': 0,
                'bounds': np.array([
                    [-100, 100],
                    [-4, 4],
                    [-40, 40],
                    [-2, 2],
                    [0.00001, 2],
                ]),
                'death_rate': 0.03,
                'gamma_scale': 3.64,
                'gamma_shape': 6.28,
                'period': 7,
                'prediction_upper_bound': 5000}


class ArmadilloV1:
    """
    An implementation of Larry and Valerie's mobility model 

    Attributes:
        args (dict): A dictionary containing values for A, alpha, beta, mu, sig.
          It also will contain the M and DC used for latest fit.
        preds (numpy array): Best predictions stored from latest fit
        fit: A method for fitting model parameters
        eval: A method for evaluating the model at timestep t
    """

    def __init__(self, args={}):
        """
        Inits model parameters

        Args: 
            args (dict) : args for model training/tuning, default empty. Required
              parameters for the model includes 'ALPHA', 'BETA', 'MU', 'bounds',
              'death_rate', 'gamma_scale', 'gamma_shape' and 'period'. If not provided
              in args, default values will be used.
        """

        self.args = args

        for key in DEFAULT_ARGS.keys():
            if args.get(key, None) is None:
                self.args[key] = DEFAULT_ARGS[key]

    def fit(self, M, DC, y_true, loss=hellinger, initial_temp=20000, visit=2.0):
        """
        Method for fitting model parameters. The mobility time series M and
        death curve DC are also saved to args. Best predictions is saved to the
        preds variable.

        Args:
            M (numpy array, shape=(num_intervals)): mobility time series
            DC (numpy array, shape=(num_intervals)): death curve
            y_true: observed values
            optimizer: optimizer for model
            loss: loss function
            initial_temp (float): initial temperature for dual annealing
            visit (float): another parameter for dual annealing
        """
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

        par_init = np.array(
            [1.5+np.mean(y_true)/L, self.args['ALPHA'], self.args['BETA'], self.args['MU'], 1])

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
        out[out > self.args['prediction_upper_bound']
            ] = self.args['prediction_upper_bound']
        out[out < 0] = 0

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

        # TODO: Vary t in a nonlinear way
        t = np.linspace(start=0, stop=l+L, num=l+L+1)
        # a - shape parameter #TODO: Adaptively change scale and a as new data is ingested
        DC = gamma.pdf(
            t*self.args['period'], scale=self.args['gamma_scale'], a=self.args['gamma_shape'])
        DC = (DC/np.sum(DC)) * self.args['death_rate']

        return self._eval(M, DC[:l+L], l+L, self.args["A"], self.args["alpha"], self.args["beta"], self.args["mu"], self.args["sig"])


class ArmadilloV2(tf.keras.Model):
    """
    An implementation of Larry and Valerie's mobility model in Tensorflow with
    additional methods for producing It trajectories

    Attributes:
        A (tf.Variable(float)):
        scale_correction (float): Scaling correction for fitting case counts
        shape (float): Scale 
        scale (float): Shape parameter for Gamma distribution
        DC (array_like): Probability distribution for likelihood of dying at
          time t if infected at t=0.
        lag (int): lag time between 
        num_sigmoids: Number of sigmoids in the first layer of the model
    """

    def __init__(self, scale=3.64, shape=6.28, corr="positive"):
        super(ArmadilloV2, self).__init__()
        self.corr = corr

        if corr == "positive":
            self.alpha = tf.Variable(.5, dtype=dtype, name="alpha")
            self.beta = tf.Variable(-.5, dtype=dtype, name="beta")
        else:
            self.alpha = tf.Variable(-.5, dtype=dtype, name="alpha")
            self.beta = tf.Variable(.5, dtype=dtype, name="beta")

        self.A = tf.Variable(0.1, dtype=dtype, name="A")
        self.scale = 1.0
        self.shape = None
        self.a = None
        self.DC = None
        self.lag = tf.Variable(0, name="lag")
        self.num_sigmoids = 10

        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.dense1 = tf.keras.layers.Dense(
            self.num_sigmoids,
            activation=None,
            kernel_initializer='ones',
            use_bias=False,
            name="dense1")
        self.dense1.trainable = False

        self.linear2 = LinearLayer("linear2")
        self.dense3 = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer='ones',
            use_bias=False,
            name="dense3")
        self.dense3.trainable = False

        self.compile_DC()

    def compile_DC(self, DC=None, scale=3.64, shape=6.28, l=700, interval=1, aggregation='sum'):
        OPTIONS = ["first", "last", "sum"]
        assert aggregation in OPTIONS
        self.scale = scale
        self.shape = shape
        t = tf.range(l*interval)
        DC = gamma.pdf(t, scale=scale, a=shape)  # a - shape parameter
        DC = tf.reshape(DC, shape=(l, interval))
        if aggregation == "first":
            DC = DC[:, 0]
        elif aggregation == "last":
            DC = DC[:, -1]
        elif aggregation == "sum":
            DC = tf.reduce_sum(DC, axis=1)
        DC = (DC/sum(DC)) * 0.03
        self.DC = tf.cast(DC, dtype)

    def eval_g(self, M):
        g = self.dense1(M)
        g = self.linear2(g)
        g = self.sigmoid(g)
        g = self.dense3(g)/self.num_sigmoids
        return g

    def eval_power(self, M):
        g = self.eval_g(M)
        return self.beta + self.alpha * g

    def forecast_cases(self, M, n):
        M = tf.squeeze(M)
        M = tf.concat([M, tf.repeat(M[-1], n)], axis=0)[:n]
        M = tf.reshape(M, shape=(1, -1, 1))
        Beta = self.eval_power(M)
        Beta = tf.squeeze(Beta)
        BetaSum = tf.cumsum(Beta)
        it = tf.exp(BetaSum)
        return self.scale * it

    def fit_scale(self, M, cases, cumsum=True):
        cases = tf.squeeze(cases)
        M = tf.squeeze(M)
        n = cases.shape[0]
        forecasts = self.forecast_cases(M, n)
        if cumsum:
            cases = tf.cumsum(cases)
            forecasts = tf.cumsum(forecasts)
        self.scale_correction = self.scale_correction * \
            (tf.reduce_sum(cases)/tf.reduce_sum(forecasts))

    def call(self, x):
        M = x

        L0 = M.shape[-2]
        L1 = M.shape[-1]

        M = tf.reshape(M, shape=(L0, 1))

        Beta = self.eval_power(M)
        Beta = tf.concat(
            [tf.zeros(shape=(1, L1), dtype=dtype), Beta], axis=0)
        BetaSum = tf.cumsum(Beta)

        BetaSum = BetaSum[:L0]

        inputs = tf.concat([tf.zeros(shape=(L0-1, L1), dtype=dtype),
                            tf.exp(BetaSum[:L0-1])], axis=0)
        inputs = tf.reshape(inputs, shape=(1, inputs.shape[0], 1), name="DC0")

        filters = self.DC[:L0-1][::-1]
        filters = tf.reshape(filters, shape=(
            filters.shape[0], 1, 1), name="kernel")
        outputs = tf.nn.conv1d(inputs, filters, stride=1, padding='VALID')
        outputs = tf.squeeze(outputs, name='outputs')
        return self.A*outputs

    def train_step(self, data):
        M, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self(M, training=True)
            loss = self.loss(y_true, y_pred)
        variables = [
            var for var in self.trainable_variables if var.name[:3] != "lag"]
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(
            zip(gradients, [var for var in self.trainable_variables if var.name[:3] != "lag"]))

        if self.corr == "positive":
            self.alpha.assign(tf.clip_by_value(self.alpha, 1e-5, 5.0))
            self.beta.assign(tf.clip_by_value(self.alpha, -5.0, -1e-5))
        else:
            self.alpha.assign(tf.clip_by_value(self.alpha, -5.0, -1e-5))
            self.beta.assign(tf.clip_by_value(self.beta, 1e-5, 5.0))
        self.A.assign(tf.clip_by_value(self.A, 1e-5, 1e5))

        y_pred = self(M)
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}
