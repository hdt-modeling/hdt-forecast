import hdt_util
from datetime import date, timedelta
import covidcast

import numpy as np

class Basic_tester:
    
    def __init__(self, cache_loc):
        self.cache_loc = cache_loc
        self.feeder = hdt_util.Basic_feeder(cache_loc)
        
    def test_case_loading(self, source, signal, start_date, end_date, level, count, cumulated):
    
        cases = self.feeder.query_cases(source=source, 
                                        signal=signal, 
                                        start_date=start_date,
                                        end_date=end_date, 
                                        level=level, 
                                        count=count, 
                                        cumulated=cumulated)
        
        if cases is not None:
            new_loader = hdt_util.CovidcastGetter(self.cache_loc)
            #check cases
            signal_name = signal
            if cumulated:
                signal_name += '_cumulative'
            else:
                signal_name += '_incidence'
            if count:
                signal_name += '_num'
            else:
                signal_name += '_prop'
            direct_cases = new_loader.query(data_source=source,
                                            signal=signal_name,
                                            forecast_date=end_date,
                                            geo_type=level,
                                            start_date=start_date)
            
            cases.dropna(inplace=True)
            direct_cases.dropna(inplace=True)
            
            assert direct_cases.shape[0] == cases.shape[0]
            assert direct_cases.shape[1] == cases.shape[1]
            
            for i in range(cases.shape[0]):
                for j in range(cases.shape[1]):
                    val1 = direct_cases.iloc[i, j]
                    val2 = cases.iloc[i, j]
                    assert  val1 == val2, 'at location {}, {} != {}'.format((i, j), val1, val2)
                        
    def test_mobility_loading(self, signal, start_date, end_date, geo_type):
        
        mobility = self.feeder.query_mobility(signal, start_date, end_date, geo_type)
        new_loader = hdt_util.CovidcastGetter(self.cache_loc)
        
        if isinstance(signal, int):
            signal = self.feeder.safegraph_keys[signal]
        direct_mobility = new_loader.query(data_source='safegraph',
                                           signal=signal,
                                           start_date=start_date,
                                           forecast_date=end_date,
                                           geo_type=geo_type)
        
        mobility.dropna(inplace=True)
        direct_mobility.dropna(inplace=True)
        
        assert direct_mobility.shape[0] == mobility.shape[0]
        assert direct_mobility.shape[1] == mobility.shape[1]
        for i in range(mobility.shape[0]):
            for j in range(mobility.shape[1] - 1):
                val1 = direct_mobility.iloc[i, j]
                val2 = mobility.iloc[i, j]
                assert  val1 == val2, 'at location {}, {} != {}'.format((i, j), val1, val2)
                    
    def test_leading_indicator_loading(self, source, signal, start_date, end_date, level):
        
        indicator = self.feeder.query_leading_indicator(source, signal, start_date, end_date, level)
        new_loader = hdt_util.CovidcastGetter(self.cache_loc)
        
        direct_indicator = new_loader.query(data_source=source, 
                                            signal=signal, 
                                            start_date=start_date, 
                                            forecast_date=end_date, 
                                            geo_type=level)
        
        indicator.dropna(inplace=True)
        direct_indicator.dropna(inplace=True)
        
        assert direct_indicator.shape[0] == indicator.shape[0]
        assert direct_indicator.shape[1] == indicator.shape[1]
        for i in range(indicator.shape[0]):
            for j in range(indicator.shape[1]):
                val1 = direct_indicator.iloc[i, j]
                val2 = indicator.iloc[i, j]
                assert  val1 == val2, 'at location {}, {} != {}'.format((i, j), val1, val2)
                
if __name__ == '__main__':
    
    tester = Basic_tester('test_data/request_cache')
    print('Testing count case loading')
    tester.test_case_loading('jhu-csse', 'confirmed', date(2020, 9, 1), date(2020, 10, 1), 'state', True, False)
    print('Testing death case loading')
    tester.test_case_loading('usa-facts', 'deaths', date(2020, 9, 1), date(2020, 10, 1), 'state', True, False)
    print('Testing mobility loading')
    tester.test_mobility_loading( 1, date(2020, 9, 1), date(2020, 10, 1), 'state')
    print('Testing leading indicator loading')
    tester.test_leading_indicator_loading('fb-survey', 'smoothed_cli', date(2020, 9, 1), date(2020, 10, 1), 'state')
    print('Data loading finished')
    print('Passed All Tests')