# a parallelized implementation of Larry and Valerie's mobility model
from scipy.stats import gamma, norm
import numpy as np
import pandas
import math
import os
import multiprocessing as mp
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pyplot
from forecasters.lv_mobility import LVMM
from plots import plot_fit


def worker(i, M, DC, y_true):
    model = LVMM()
    model.fit(
        M=M,
        DC=DC,
        y_true=y_true,
    )
    return i, model.args


def error_callback(exception):
    print(exception)


def callback(args):
    keys = ['A', 'alpha', 'beta', 'mu', 'sig']
    for j, key in enumerate(keys):
        theta[args[0], j] = args[1][key]


if __name__ == "__main__":
    # TODO: Should be replaced with Lifeng's data pipeline
    # Some of the features we may want in the pipeline:
    # The ability to choose the start and end dates
    # The option to break up the data by a set number, e.g. every 7 days
    # The ability to transform data in these grouped intervals with user defined transformations
    def we(a):
        n = len(a)
        b = math.floor(n/7)
        nn = b*7
        a = a[0:nn]
        a = a.values.reshape((b, 7))
        a = np.sum(a, axis=1)
        a = a.reshape((1, b))
        return a

    df = pandas.read_csv(os.path.join(os.getcwd(), "data/state_full.csv"))
    df = df[df["state"] != "pr"]

    date = df["date"]
    deaths = df["ndeaths"]
    home = df["completely_home_prop"]
    work = df["full_time_work_prop"]
    cases = df["ncases"]
    state = df["state"]
    State = pandas.unique(state)
    deaths[deaths < 0] = 0

    Y = A = C = None
    n = len(deaths)
    for i in range(51):
        if i == 0:
            I = (state == State[i])
            Y = we(deaths[I])
            A = we(home[I])  # using HOME
            C = we(cases[I])
        else:
            I = (state == State[i])
            Y = np.concatenate([Y, we(deaths[I])])
            A = np.concatenate([A, we(home[I])])  # using HOME
            C = np.concatenate([C, we(cases[I])])
    #######################################################

    l = 25
    t = np.linspace(start=0, stop=l, num=l+1)
    ft = gamma.pdf(t*7, scale=3.64, a=6.28)  # a - shape parameter
    ft = (ft/sum(ft)) * 0.03
    x = range(1, l+1)

    pdf = PdfPages("plots/fit_optim.pdf")
    _, axs = pyplot.subplots(3, 3, figsize=(8, 8))
    theta = np.zeros((51, 5))
    pool = mp.Pool(mp.cpu_count())

    # training loop
    for i in range(51):
        y_true = Y[i, :l]
        m = A[i, :l]
        pool.apply_async(
            worker,
            [i, m, ft[:l], y_true],
            callback=callback,
            error_callback=error_callback,
        )
    pool.close()
    pool.join()

    # plotting predictions and observations for State[i]
    for i in range(51):
        y_true = Y[i, :l]
        m = A[i, :l]
        model = LVMM()
        y_pred = model._eval(
            M=m,
            DC=ft[:l],
            L=l,
            A=theta[i, 0],
            alpha=theta[i, 1],
            beta=theta[i, 2],
            mu=theta[i, 3],
            sig=theta[i, 4],
        )
        if i % 9 == 0:
            _, axs = pyplot.subplots(3, 3, figsize=(8, 8))

        axs = plot_fit(pdf, i, x, y_true, y_pred, State[i], axs)

        if (i+1) % 9 == 0:
            pyplot.tight_layout()
            pdf.savefig(bbox_inches="tight")
            pyplot.close()

    pyplot.tight_layout()
    pdf.savefig(bbox_inches="tight")
    pyplot.close()

    # plot parameters
    titles = ['alpha', 'beta', 'sigma', 'mu']
    _, axs = pyplot.subplots(2, 2, figsize=(8, 8))
    for i in range(4):
        axs[i//2, i % 2].scatter(range(1, 52), theta[:, i+1])
        axs[i//2, i % 2].set_title(titles[i], fontsize=10)
    pyplot.tight_layout()
    pdf.savefig(bbox_inches="tight")
    pyplot.close()

    # plot reproduction numbers
    x = np.linspace(0.1, 0.9, num=200)
    pyplot.figure(figsize=(8, 8))
    for i in range(51):
        y = np.exp(theta[i, 2]+theta[i, 1]*norm.cdf(x,
                                                       loc=abs(theta[i, 3]), scale=abs(theta[i, 4])))
        pyplot.plot(x, y)
    pyplot.xlabel("Mobility proportion", fontsize=10)
    pyplot.ylabel("Reproduction Number", fontsize=10)
    pyplot.tight_layout()
    pdf.savefig(bbox_inches="tight")
    pdf.close()
