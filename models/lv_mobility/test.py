# an implementation of Larry and Valerie's mobility model
from scipy.stats import gamma, norm
import numpy
import pandas
import math
import os
import optimizer
import multiprocessing as mp
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pyplot
from model import lv_mobility
from loss import mse, l1, hellinger
from plots import plot_fit


if __name__ == "__main__":
    # TODO: Should be replaced with Lifeng's data pipeline
    # Some of the features we may want in the pipeline:
    # The ability to choose the start and end dates
    # The option to break up the data by a set number, e.g. every 7 days and then applying a transformation to that data
    # The ability to transform data in these grouped intervals with user defined transformations
    def we(a):
        n = len(a)
        b = math.floor(n/7)
        nn = b*7
        a = a[0:nn]
        a = a.values.reshape((b, 7))
        a = numpy.sum(a, axis=1)
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
    for i in range(0, 51):
        if i == 0:
            I = (state == State[i])
            Y = we(deaths[I])
            A = we(home[I])  # using HOME
            C = we(cases[I])
        else:
            I = (state == State[i])
            Y = numpy.concatenate([Y, we(deaths[I])])
            A = numpy.concatenate([A, we(home[I])])  # using HOME
            C = numpy.concatenate([C, we(cases[I])])
    #######################################################

    l = 25
    t = numpy.linspace(start=0, stop=l, num=l+1)
    ft = gamma.pdf(t*7, scale=3.64, a=6.28)  # a - shape parameter
    ft = (ft/sum(ft)) * 0.03
    x = range(1, l+1)

    model = lv_mobility(args={})
    pdf = PdfPages("plots/fit_optim.pdf")
    _, axs = pyplot.subplots(3, 3, figsize=(8, 8))
    theta = numpy.zeros((51, 5))

    # training loop
    for i in range(51):
        model = lv_mobility(args={})
        y_true = Y[i, :l]
        m = A[i, :l]

        model.fit(
            M=m,
            DC=ft[:25],
            y_true=y_true,
            optimizer=optimizer.optim,
            loss=l1,
            args=None
        )
        theta[i, :] = list(model.args.values())

        y_pred = model._eval(
            M=m,
            DC=ft[:l],
            L=l,
            A=model.args["A"],
            alpha=model.args["alpha"],
            beta=model.args["beta"],
            mu=model.args["mu"],
            sig=model.args["sig"],
        )

        # plotting predictions and observations for State[i]
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
    x = numpy.linspace(0.1, 0.9, num=200)
    pyplot.figure(figsize=(8, 8))
    for i in range(51):
        pyplot.plot(x, theta[i, 2]+theta[i, 1]*norm.cdf(x,
                                                        loc=abs(theta[i, 3]), scale=abs(theta[i, 4])))
    pyplot.xlabel("Mobility proportion", fontsize=10)
    pyplot.ylabel("Reproduction Number", fontsize=10)
    pyplot.tight_layout()
    pdf.savefig(bbox_inches="tight")
    pdf.close()
