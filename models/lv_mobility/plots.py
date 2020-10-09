import numpy
from scipy.stats import gamma, norm
import math
import itertools
import matplotlib.pyplot as pyplot


def plot_fit(pdf, i, x, y_true, y_pred, title="", axs=None):
    """This method is used for plotting the fitted models"""
    axs[(i//3) % 3, i % 3].scatter(x, y_true, s=20,
                                   facecolors="none", edgecolors="r")
    axs[(i//3) % 3, i % 3].plot(x, y_pred,
                                linestyle="--", markersize=12, markerfacecolor="b")
    axs[(i//3) % 3, i % 3].set_title(title)
    axs[(i//3) % 3, i % 3].set_xticks(range(0, len(x), 5))
    axs[(i//3) % 3, i % 3].xaxis.set_label_text("Week", fontsize=10)
    axs[(i//3) % 3, i % 3].yaxis.set_label_text("Death count", fontsize=10)

    return axs
