from forecasters import AR
import numpy as np


def test_AR():
    model = AR(7)
    leading_indicator = np.random.uniform(100, 0, 1000)
    cases = np.random.uniform(100, 0, 1000)
    model.fit(leading_indicator)
    model.fit_scale(leading_indicator, cases)
    forecasts = model.forecast(leading_indicator, n=10)
    assert len(forecasts) == 10
