import hdt_util
from datetime import date
import covidcast

class Valerie_and_Larry_tester:
    
    def __init__(self, cache_loc, merge=False):
        self.feeder = hdt_util.Valerie_and_Larry_feeder(cache_loc, merge)
        self.merge = merge
        
    def test_data_loading(self, source, signal, start_date, end_date, level, count, cumulated, mobility_level):
    
        cases, mobility = self.feeder.get_data(source=source, 
                                               signal=signal, 
                                               start_date=start_date,
                                               end_date=end_date, 
                                               level=level, 
                                               count=count, 
                                               cumulated=cumulated,
                                               mobility_level=mobility_level)
        
        if cases is not None:
            new_loader = hdt_util.CovidcastGetter(cache_loc)
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
            direct_cases = direct_cases[['geo_value', 'time_value', 'value']]
            assert direct_cases.shape == cases.shape, 'Shapes should be the same, but encountered {} and {}'.format(direct_cases.shape, cases.shape)
            for i in range(cases.shape[0]):
                for j in range(cases.shape[1]):
                    val1 = direct_cases.iloc[i, j]
                    val2 = cases.iloc[i, j]
                    assert  val1 == val2, 'at location {}, {} != {}'.format((i, j), val1, val2)
            
            #check mobility
            direct_mobility = new_loader.query(data_source='safegraph',
                                               signal=feeder.safegraph_keys[mobility_level],
                                               start_date=start_date,
                                               forecast_date=end_date,
                                               geo_type=level)
            direct_mobility = direct_mobility[['geo_value', 'time_value', 'value']]
            assert direct_mobility.shape == mobility.shape
            for i in range(mobility.shape[0]):
                for j in range(mobility.shape[1]):
                    val1 = direct_mobility.iloc[i, j]
                    val2 = mobility.iloc[i, j]
                    assert  val1 == val2, 'at location {}, {} != {}'.format((i, j), val1, val2)
                    
    def test_data_processing(self):
        print('Processing details are still under discussing. The test is passed for now')

if __name__ == '__main__':
    cache_loc = 'test_data/request_cache'
    source = 'jhu-csse'
    signal = 'deaths'
    start_date=date(2020, 9, 1)
    end_date=date(2020, 10, 1) 
    level='state'
    count=True
    cumulated=False
    mobility_level=1
    
    tester = Valerie_and_Larry_tester(cache_loc)
    tester.test_data_loading(source, signal, start_date, end_date, level, count, cumulated, mobility_level)
    tester.test_data_processing()
    
    print('Passed All Tests')

    