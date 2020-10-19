import pytest
import numpy as np
from forecasters.lv_mobility import LVMM


def test_LVMM_eval(l=-1, args={}):
    l = 16  # length of interval model trained on
    if not args:
        args = {'A': 0.29896900277215616,
                'alpha': -1.1633246957590113,
                'beta': 1.098842537996063,
                'mu': -1.668444429302781,
                'sig': 1e-05,
                'M': [1.46104406, 1.5200612, 1.61381331, 1.52605252, 1.62909915,
                      1.41461628, 1.38812132, 1.70223098, 2.13268644, 2.09180085,
                      2.35881594, 2.2888028, 2.12219974, 2.04254759, 1.97069726,
                      1.87603477],
                'DC': [0.00000000e+00, 1.36714678e-03, 7.76375505e-03, 9.65277041e-03,
                       6.44387059e-03, 3.05949665e-03, 1.17096864e-03, 3.86226989e-04,
                       1.14250577e-04, 3.11001172e-05, 7.92829643e-06, 1.91668861e-06,
                       4.43499839e-07, 9.89135279e-08, 2.13800701e-08, 4.49814289e-09]}

    model = LVMM(args=args)

    predictions = model._eval(
        M=model.args['M'],
        DC=model.args['DC'],
        L=l,
        A=model.args['A'],
        alpha=model.args['alpha'],
        beta=model.args['beta'],
        mu=model.args['mu'],
        sig=model.args['sig']
    )
    assert predictions.shape[0] == 16


def test_LVMM_forecast(l=-1, args={}):
    l = 16  # length of interval model trained on

    args = {'A': 0.29896900277215616,
            'alpha': -1.1633246957590113,
            'beta': 1.098842537996063,
            'mu': -1.668444429302781,
            'sig': 1e-05,
            'M': [1.46104406, 1.5200612, 1.61381331, 1.52605252, 1.62909915,
                  1.41461628, 1.38812132, 1.70223098, 2.13268644, 2.09180085,
                  2.35881594, 2.2888028, 2.12219974, 2.04254759, 1.97069726,
                  1.87603477],
            'DC': [0.00000000e+00, 1.36714678e-03, 7.76375505e-03, 9.65277041e-03,
                   6.44387059e-03, 3.05949665e-03, 1.17096864e-03, 3.86226989e-04,
                   1.14250577e-04, 3.11001172e-05, 7.92829643e-06, 1.91668861e-06,
                   4.43499839e-07, 9.89135279e-08, 2.13800701e-08, 4.49814289e-09]}
    model = LVMM(args=args)

    l2 = 25  # length of interval we want to forecast up to

    forecast = model.forecast(l2)

    assert forecast.shape[0] == 25

    forecast = model.forecast(l2, M=np.array(args['M']))

    assert forecast.shape[0] == 25

    forecast = model.forecast(l2, M=np.array(args['M']), DC=np.array(args['DC']))

    assert forecast.shape[0] == 25

