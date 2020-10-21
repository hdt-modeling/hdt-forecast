import numpy as np
from scipy.optimize import dual_annealing
import itertools


def optim(model, M, DC, y_true, loss, args=None):
    """
    A method for constrained optimization using dual_annealing from scipy.optimize for lv_mobility.

    Args:
        M (numpy array): mobility time series
        DC (numpy array): death curve
        y_true (numpy array): observed value of cases
        loss: loss function 
        args (dict): ALPHA, BETA, MU, array of parameters for initialization
    """
    if args == None:
        ALPHA = np.linspace(-2, 2, num=6)
        BETA = np.linspace(-2, 2, num=6)
        MU = np.linspace(0.1, 0.5, num=5)
        bounds = np.array([
            [-100, 100],
            [-4, 4],
            [-40, 40],
            [-2, 2],
            [0.00001, 2],
        ])
    else:
        ALPHA = args["ALPHA"]
        BETA = args["BETA"]
        MU = args["MU"]
        if args.has_key("bounds"):
            bounds = args["bounds"]
        else:
            bounds = np.array([
                [-100, 100],
                [-4, 4],
                [-40, 40],
                [-2, 2],
                [0.00001, 2],
            ])

    iterator = itertools.product(
        range(len(ALPHA)),
        range(len(BETA)),
        range(len(MU)),
    )

    def loss_wrap(x, *args):
        y_pred = model._eval(M, DC, L, x[0], x[1], x[2], x[3], x[4])
        return loss(y_true, y_pred)

    best_loss = float("inf")
    L = len(M)

    for i, j, k in iterator:
        par_init = np.array(
            [1.5+np.mean(y_true)/L, ALPHA[i], BETA[j], MU[k], 1])

        args_ = [M, DC, L]
        res = dual_annealing(
            func=loss_wrap,
            args=args_,
            bounds=bounds,
            x0=par_init,
            initial_temp=20000,
            visit=2.0,
            maxiter=200,
        )

        if res["fun"] < best_loss:
            best_loss = res["fun"]
            model.args["A"] = res["x"][0]
            model.args["alpha"] = res["x"][1]
            model.args["beta"] = res["x"][2]
            model.args["mu"] = res["x"][3]
            model.args["sig"] = res["x"][4]
