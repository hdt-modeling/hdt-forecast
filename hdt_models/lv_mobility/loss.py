import numpy


def mse(y_true, y_pred):
    return numpy.mean((y_true-y_pred) ** 2)


def l1(y_true, y_pred):
    return numpy.sum(abs(y_true-y_pred))


def hellinger(y_true, y_pred):
    return numpy.sum((y_true**0.5-y_pred**0.5)**2)
