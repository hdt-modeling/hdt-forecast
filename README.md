# Healthy Davis Together (HDT) Forecasting Efforts

## Structure

We built two packages. `forecasters` and `hdt_util`.

`forecasters` contains the models we built to predict death cases and confirmed cases. For more details, please check `forecasters` folder.

`hdt_util` contains other parts to build the whole train-tune-predict pipeline. For more details, please check `hdt_util` folder.

## Installation

We suggest installing our package in a new virtual envirionment. You can learn about how to create a new virtual envirionment with Anaconda at [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Go to the directory containing theses files and run the following commands to install our package.

Install the utilities package with 
```
pip install -e hdt_util
```

Install the forecasters package with

```
pip install -e forecasters
```

Run the tests for `hdt_util` with
```
cd hdt_util/tests
pytest
```
