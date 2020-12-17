import covidcast
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf

from hdt_util.weekday import dow_adjust_cases
from hdt_util.delay import Delay
from hdt_util.conv1d import *


class TestDelay:
    #This is mostly code copied from Maria's notebooks
    fl_line_data = "tests/test_data/FL_line_list.csv"
    us_zip_data_path = "tests/test_data/02_20_uszips.csv"

    florida_df = pd.read_csv(fl_line_data, parse_dates=[
                             "Case_", "EventDate", "ChartDate"])
    florida_delays = (florida_df.ChartDate - florida_df.EventDate).dt.days
    florida_delays = florida_delays[florida_delays.gt(
        0) & florida_delays.lt(60)]
    fl_delay_dist = Delay.get_delay_distribution(florida_delays)
    start_date = datetime(2020, 4, 15)
    end_date = datetime(2020, 7, 15)

    cases_df = covidcast.signal(
        'indicator-combination',
        'confirmed_7dav_incidence_num',
        start_date,
        end_date,
        geo_type='county',
    )

    cumulative_cases_df = covidcast.signal(
        'indicator-combination',
        'confirmed_7dav_cumulative_num',
        end_date,
        end_date,
        geo_type='county',
    )

    thresh_geos = cumulative_cases_df[cumulative_cases_df.value > 500].geo_value

    # get all florida fips codes
    geo_map = pd.read_csv(
        us_zip_data_path,
        usecols=["fips", "state_id", "population"],
        dtype={"state_id": str},
        converters={"fips": lambda x: str(x).zfill(5)},
    )

    florida_geo = geo_map[geo_map.state_id.eq("FL")]
    florida_population = florida_geo.groupby(
        "fips").population.sum().reset_index()
    florida_fips = florida_geo.fips.unique()

    cases_df = cases_df.set_index(["geo_value"])
    geos = cases_df.index.unique()
    geos = geos[geos.isin(florida_fips)]  # only keep florida geos
    geos = geos[geos.isin(thresh_geos)]  # counties with >500 cumulative cases

    def test_deconv(self):
        delay_dist = self.fl_delay_dist
        
        geo = "12086"
        cases = self.cases_df.loc[geo].sort_values(by='time_value')
        n = cases.value.shape[0]
        train_time = cases.time_value[:]
        train_cases_ = cases[cases.time_value.isin(train_time)]

        # dow_adjust and deconvolution will both be done in Delay.deconv
        train_cases = dow_adjust_cases(train_cases_, lam=10)
        sub_infections = np.clip(admm_deconvolution_v2(
            train_cases, delay_dist, 3000, 3000, n_iters=500, k=2), 0, np.inf)

        assert np.array_equal(sub_infections, Delay.deconv(train_cases_, delay_dist)), "The two deconvolved arrays do not match"

    def test_conv(self):
        x = tf.random.uniform((100,1), minval=0, maxval=100)
        Delay.conv(x, self.fl_delay_dist)
