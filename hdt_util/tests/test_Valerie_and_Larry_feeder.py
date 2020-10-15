import hdt_util
from datetime import date, timedelta
import covidcast

import numpy as np

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
            direct_cases.rename({'value':'case_value', 'time_value':'date'}, axis=1, inplace=True)
            direct_cases.reset_index(inplace=True, drop=True)
            assert direct_cases.shape[0] == cases.shape[0], 'First dimension of shapes should be the same, but encountered {} and {}'.format(direct_cases.shape, cases.shape)
            for i in range(cases.shape[0]):
                for j in direct_cases.columns:
                    val1 = direct_cases.loc[i, j]
                    val2 = cases.loc[i, j]
                    assert  val1 == val2, 'at location {}, {} != {}'.format((i, j), val1, val2)
            
            #check mobility
            direct_mobility = new_loader.query(data_source='safegraph',
                                               signal=self.feeder.safegraph_keys[mobility_level],
                                               start_date=start_date,
                                               forecast_date=end_date,
                                               geo_type=level)
            direct_mobility = direct_mobility[['geo_value', 'time_value', 'value']]
            assert direct_mobility.shape[0] == mobility.shape[0]
            assert direct_mobility.shape[1] == mobility.shape[1] - 1
            for i in range(mobility.shape[0]):
                for j in range(mobility.shape[1] - 1):
                    val1 = direct_mobility.iloc[i, j]
                    val2 = mobility.iloc[i, j]
                    assert  val1 == val2, 'at location {}, {} != {}'.format((i, j), val1, val2)
                    
    def test_avg_pooling(self, source, signal, start_date, end_date, level, count, cumulated, mobility_level, period):
        cases, mobility = self.feeder.get_data(source=source, 
                                               signal=signal, 
                                               start_date=start_date,
                                               end_date=end_date, 
                                               level=level, 
                                               count=count, 
                                               cumulated=cumulated,
                                               mobility_level=mobility_level)
        
        avg_cases = self.feeder.average_pooling(cases, period, end_date+timedelta(days=10))
        avg_mobility = self.feeder.average_pooling(mobility, period, end_date+timedelta(days=10))
        
        end_date = end_date + timedelta(days=10)
        max_time = avg_cases['time'].max()
        for geo in avg_cases['geo_value'].unique():
            for time in avg_cases['time'].unique():
                temp_start_date = end_date - timedelta(days=int(period*(max_time-time) + period - 1))
                temp_end_date = end_date - timedelta(days=int(period*(max_time-time)))
                temp_cases = cases[cases['date'].apply(lambda x : temp_start_date <= x <= temp_end_date)]
                temp_mobility = mobility[mobility['date'].apply(lambda x : temp_start_date <= x <= temp_end_date)]
                
                if temp_cases.shape[0] > 0:
                    value1 = avg_cases[(avg_cases['geo_value']==geo)&(avg_cases['time']==time)]['case_value'].values[0]
                    value2 = np.mean(temp_cases[temp_cases['geo_value']==geo]['case_value'])
                    assert abs(value1 - value2) < 1e-4, '{}, {}, {}'.format(value1, value2, abs(value1-value2))
                else:
                    assert avg_cases[(avg_cases['geo_value']==geo)&(avg_cases['time']==time)].isnull()['case_value'].values[0]
                
                if temp_mobility.shape[0] > 0:
                    value1 = avg_mobility[(avg_mobility['geo_value']==geo)&(avg_mobility['time']==time)]['mobility_value'].values[0]
                    value2 = temp_mobility[temp_mobility['geo_value']==geo]['mobility_value'].values
                    n = value2.shape[0]
                    value2 = n/period*np.mean(value2) + (period-n)/period*value2[-1]
                    assert abs(value1 - value2) < 1e-4, '{}, {}, {}'.format(value1, value2, abs(value1-value2))
                else:
                    value1 = avg_mobility[(avg_mobility['geo_value']==geo)&(avg_mobility['time']==time)]['mobility_value'].values[0]
                    value2 = mobility[(mobility['geo_value']==geo)&(mobility['date']==mobility['date'].max())]['mobility_value'].values[0]
                    assert abs(value1 - value2) < 1e-4, '{}, {}, {}'.format(value1, value2, abs(value1-value2))
        
    def test_area_select(self, source, signal, start_date, end_date, level, count, cumulated, mobility_level, areas):
        cases, mobility = self.feeder.get_data(source=source, 
                                               signal=signal, 
                                               start_date=start_date,
                                               end_date=end_date, 
                                               level=level, 
                                               count=count, 
                                               cumulated=cumulated,
                                               mobility_level=mobility_level)
        
        areas = [val.lower() for val in areas]
        filtered_cases = self.feeder.area_filter(cases, areas)
        filtered_mobility = self.feeder.area_filter(mobility, areas)
        
        for i in range(filtered_cases.shape[0]):
            assert filtered_cases.loc[i, 'geo_value'].lower() in areas, '{} not in {}'.format(filtered_cases.loc[i, 'geo_value'].lower(), areas)
        for i in range(filtered_mobility.shape[0]):
            assert filtered_mobility.loc[i, 'geo_value'].lower() in areas, '{} not in {}'.format(filtered_mobility.loc[i, 'geo_value'].lower(), areas)

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
    period_choices = [3, 4, 7]
    areas = ['CA', 'WA', 'AK']
    
    tester = Valerie_and_Larry_tester(cache_loc)
    print('Testing data loading')
    tester.test_data_loading(source, signal, start_date, end_date, level, count, cumulated, mobility_level)
    print('Data loading finished')
    for period in period_choices:
        print('Testing period {}'.format(period))
        tester.test_avg_pooling(source, signal, start_date, end_date, level, count, cumulated, mobility_level, period)
        print('Period {} finished'.format(period))
    print('Testing selection of area')
    tester.test_area_select(source, signal, start_date, end_date, level, count, cumulated, mobility_level, areas)
    print('Area selection finished')
    print('Passed All Tests')

    