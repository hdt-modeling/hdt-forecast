import hdt_util
from datetime import date
import covidcast

import numpy as np

class ARLIC_tester:
    
    def __init__(self, cache_loc):
        self.cache_loc = cache_loc
        self.feeder = hdt_util.ARLIC_feeder(cache_loc)
        
    def test_data_loading(self, case_source, case_signal, li_source, li_signal, start_date, end_date, level):
    
        data = self.feeder.get_data(case_source=case_source, 
                                     case_signal=case_signal, 
                                     li_source=li_source, 
                                     li_signal=li_signal, 
                                     start_date=start_date,
                                     end_date=end_date, 
                                     level=level)
        
        if data is not None:
            new_loader = hdt_util.CovidcastGetter(self.cache_loc)
            #check cases
            signal_name = case_signal
            direct_cases = new_loader.query(data_source=case_source,
                                            signal=signal_name,
                                            forecast_date=end_date,
                                            geo_type=level,
                                            start_date=start_date)
            direct_cases = direct_cases[['geo_value', 'time_value', 'value']]
            direct_cases.rename({'value':'case_value', 'time_value':'date'}, axis=1, inplace=True)
            direct_cases.reset_index(inplace=True, drop=True)
                    
            signal_name = li_signal
            direct_li = new_loader.query(data_source=li_source,
                                            signal=signal_name,
                                            forecast_date=end_date,
                                            geo_type=level,
                                            start_date=start_date)
            direct_li = direct_li[['geo_value', 'time_value', 'value']]
            direct_li.rename({'value':'li_value', 'time_value':'date'}, axis=1, inplace=True)
            direct_li.reset_index(inplace=True, drop=True)
            
            direct_data = direct_cases.merge(direct_li, on=['geo_value', 'date'], how='inner')
            assert direct_data.shape[0] == data.shape[0], 'First dimension of shapes should be the same, but encountered {} and {}'.format(direct_data.shape, data.shape)
            for i in range(data.shape[0]):
                for j in direct_data.columns:
                    val1 = direct_data.loc[i, j]
                    val2 = data.loc[i, j]
                    assert  val1 == val2, 'at location {}, {} != {}'.format((i, j), val1, val2)
        
    def test_area_select(self, case_source, case_signal, li_source, li_signal, start_date, end_date, level, areas):
        cases = self.feeder.get_data(case_source=case_source, 
                                     case_signal=case_signal, 
                                     li_source=li_source, 
                                     li_signal=li_signal, 
                                     start_date=start_date,
                                     end_date=end_date, 
                                     level=level)
        
        areas = [val.lower() for val in areas]
        filtered_cases = self.feeder.area_filter(cases, areas)
        
        for i in range(filtered_cases.shape[0]):
            assert filtered_cases.loc[i, 'geo_value'].lower() in areas, '{} not in {}'.format(filtered_cases.loc[i, 'geo_value'].lower(), areas)

if __name__ == '__main__':
    cache_loc = 'test_data/request_cache'
    case_source = 'indicator-combination'
    case_signal = 'confirmed_7dav_cumulative_num'
    li_source = 'fb-survey'
    li_signal = 'smoothed_cli'
    start_date = date(2020, 9, 1)
    end_date = date(2020, 10, 1)
    level = 'state'
    areas = ['CA', 'WA', 'AK']
    
    tester = ARLIC_tester(cache_loc)
    print('Testing data loading')
    tester.test_data_loading(case_source, case_signal, li_source, li_signal, start_date, end_date, level)
    print('Testing selection of area')
    tester.test_area_select(case_source, case_signal, li_source, li_signal, start_date, end_date, level, areas)
    print('Area selection finished')
    print('Passed All Tests')