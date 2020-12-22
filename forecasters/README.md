## HDT Models Package

## Models
1. ArmadilloV1: This model is an implementation of Larry and Valerie's mobility model for forecasting. This particular implementation can only be used for forecasting death counts.
2. ARLIC: This model uses leading indicators to forecast case counts. It does so by using two separate autoregressive models. The first autoregressive model is fitted on the leading indicator which allows it to roll this covariate into the future. The second model uses the leading indicator as a covariate which is convolved with a delay distribution and then fitted to case counts.

## Usage

After installation (follow instructions at `../README.md`) you can import models as follows, using ARLIC as an example:

```
from forecasters import ARLIC
```

Each model has a `fit` and `forecast` method. The `fit` method requires users to pass in a dictionary of relevent parameters for fitting the model. The `forecast` method currently only does forecasts for a single covariate, case counts or death counts. In the future there will be an argument included which allows you to toggle which covariate you would like to forecast with using that model. Examples for how these models are used can been in the notebooks included in `../examples`.



