# HDT Utilities Package

## Basic Data Loader

The `CovidcastGetter` class in `get_covidcast.py` is a convenient way to query data. One can choose the signal to query and from which database to query through specifying `signal` and `data_source`. One can also pick the `start_date` and `end_date` to download data. The data will be downloaded as it looks like on `end_date`. All queries will be cached.

For examples, please check files in `examples` foleder. [Here is the quick link](https://github.com/hdt-modeling/hdt-forecast/tree/master/examples)

## Convenient Data Loader

The `Basic_feeder` class in `data_feeder.py` helps one to get specific of data more conveniently. It has `query_cases`, `query_leading_indicator` and `query_mobility` to download confirmed/death cases (cumulated or not, ratio or count), leading indicators in surveys and mobility data from safegraph respectively. Its `pooling` method allows one to aggregate the data with every given number of days, `period`, to decide the time resolution for data training. There is also a `area_filter` method to filter out desired entries that belong to certain specified area(s).

Besides, `ArmadilloV1_feeder` and `ARLIC_feeder` are data loaders specially designed for our models `ArmadilloV1` and `ARLIC`.

For examples, please check files in `examples` foleder. [Here is the quick link](https://github.com/hdt-modeling/hdt-forecast/tree/master/examples)

## Backtesting/Evaluation

`evaluation.py` has backtesting/evaluating classes for our models, `ArmadilloV1_evaluator` and `ARLIC_evaluator`. One can specify the `start_date` and `end_date` for data, geometric level (state or county), potentially specify which states/counties, how many predictions to make, and which the metrics to monitor. The evaluators will return two data frames, one for prediction results and the other for evaluation metrics.

For examples, please check files in `examples` foleder. [Here is the quick link](https://github.com/hdt-modeling/hdt-forecast/tree/master/examples)

## Converter (Address to Census Tract/Block/CBG)

Use the `address_to_census` method in the `converter` module.  You can see an implementation in `tests/test_converter.py`.

```
import pandas as pd
from hdt_util.converter import address_to_census

address = "1 Shields Avenue, Davis, CA, 95616"
GEOID = address_to_census(address, aggregation="block groups")
```

## Data Processing

Still reading Maria's code to understand what it does exactly and how to modify it for easier use.
