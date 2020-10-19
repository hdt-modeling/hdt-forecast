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
from plots import plot_mobility

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
    part = df["part_time_work_prop"]
    median = df["median_home_dwell_time"]
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
            #A = we(home[I])  # using HOME
            #A = we(work[I])
            #A = we(part[I])
            A = we(median[I])
            C = we(cases[I])
        else:
            I = (state == State[i])
            Y = np.concatenate([Y, we(deaths[I])])
            #A = np.concatenate([A, we(home[I])])  # using HOME
            #A = np.concatenate([A, we(work[I])])
            #A = np.concatenate([A, we(part[I])])
            A = np.concatenate([A, we(median[I])])
            C = np.concatenate([C, we(cases[I])])
    #######################################################
    l=25
    pdf = PdfPages("plots/median_home_dwell_time.pdf")
    _, axs = pyplot.subplots(3, 3, figsize=(8, 8))
    x = range(1, l+1)
    for i in range(51):
        if i % 9 == 0:
            _, axs = pyplot.subplots(3, 3, figsize=(8, 8))
        
        y = A[i,:l]
        axs = plot_mobility(
            pdf, i, l, x, y, State[i], "median_home_dwell_time", axs)

        if (i+1) % 9 == 0:
            pyplot.tight_layout()
            pdf.savefig(bbox_inches="tight")
            pyplot.close()

    pyplot.tight_layout()
    pdf.savefig(bbox_inches="tight")
    pyplot.close()
    pdf.close()