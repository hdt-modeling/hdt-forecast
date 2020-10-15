'''
In this file, each model will have its data feeder. Class names will be the model's names 
'''

from .get_covidcast import CovidcastGetter
import pandas as pd
import numpy as np
import datetime
from datetime import date

import warnings
warnings.filterwarnings("ignore")

BASE_DATE = date(2020, 2, 29)

class Basic_feeder:
    
    def __init__(self, cache_loc=None):
        assert isinstance(cache_loc, str) or cache_loc is None, 'cache_loc should be a string'
        if cache_loc is None:
            self.data_loader = CovidcastGetter()
        else:
            self.data_loader = CovidcastGetter(cache_loc)
        
        self.safegraph_keys = {1 : 'completely_home_prop',
                               2 : 'full_time_work_prop',
                               3 : 'part_time_work_prop',
                               4 : 'median_home_dwell_time'}
        
    def query_response(self, 
                       source='jhu-csse', 
                       signal='deaths', 
                       start_date=None,
                       end_date=None, 
                       level='state', 
                       count=True, 
                       cumulated=False):
        
        #check parameters
        assert isinstance(source, str), 'source should be a string'
        source = source.lower()
        assert source in ['jhu-csse', 'usa-facts', 'indicator-combination'], 'source should be one of \'jhu-csse\', \'usa-facts\', and \'indicator-combination\''
        assert isinstance(signal, str), 'signal should be a string'
        signal = signal.lower()
        assert signal in ['deaths', 'confirmed'], 'signal should be one of \'death\' and \'confirmed\''
        assert (isinstance(start_date, datetime.date) or start_date is None) and (isinstance(end_date, datetime.date) or end_date is None), 'if start_date or end_date is provided, it must be a datetime.date object'
        assert isinstance(level, str), 'level should be a string'
        level = level.lower()
        assert level in ['state', 'county'], 'level should be one of \'state\' and \'county\''
        assert isinstance(count, bool), 'count should be True or False'
        assert isinstance(cumulated, bool), 'cumulated should be True or False'
        
        #correct values for later calls
        if end_date is None:
            end_date = date.today()
        if cumulated:
            signal += '_cumulative'
        else:
            signal += '_incidence'
        if count:
            signal += '_num'
        else:
            signal += '_prop'
        
        case_data = self.data_loader.query(data_source=source,
                                           signal=signal,
                                           start_date=start_date,
                                           forecast_date=end_date,
                                           geo_type=level)
        
        return case_data

class Valerie_and_Larry_feeder(Basic_feeder):
    '''
    Link to explanation document : https://drive.google.com/drive/u/0/folders/13i2PVMlADp_vw8VqxlhzApSGzsjILbWC
    
    This models requires 4 inputs:
        Area: the code for location
        Time: indicator of time
        Death Count: count of death cases
        Mobility: level of mobility
    
    According the data we have:
        Area must be county level or state level, default is state level, we do not have mobility data for other levels of area
        Time is on a day level
        
        Death Count will be generalized. You can:
            choose confirmed case or death case
            choose new case or cumulative case
            choose proportion (in 100000) or count
            choose from three datasets: JHU Cases, USAFacts or a combination from delphi's team
            default is new death case count from JHU Cases
            
        Mobility is obtained from SafeGraph. There are four choices:
            (default) completely_home_prop : The fraction of mobile devices that did not leave the immediate area of their home (SafeGraph’s completely_home_device_count / device_count)
            full_time_work_prop : The fraction of mobile devices that spent more than 6 hours at a location other than their home during the daytime (SafeGraph’s full_time_work_behavior_devices / device_count)
            part_time_work_prop : The fraction of devices that spent between 3 and 6 hours at a location other than their home during the daytime (SafeGraph’s part_time_work_behavior_devices / device_count)
            median_home_dwell_time : The median time spent at home for all devices at this location for this time period, in minutes
    '''
    
    def __init__(self, cache_loc=None, merge=True):
        super(Valerie_and_Larry_feeder, self).__init__(cache_loc)
        self.merge = merge
            
    def get_data(self, 
                 source='jhu-csse', 
                 signal='deaths', 
                 start_date=None,
                 end_date=None, 
                 level='state', 
                 count=True, 
                 cumulated=False,
                 mobility_level=1):
        '''
        Returns data as ordered.
        
        Params:
        =======
        source : string, optional, default 'jhu-csse', the data source from which we download data
        signal : 'deaths' or 'confirmed', optional, default 'deaths', which kind of data to download
        start_data : None or a datetime.date object, optional, default None, the start date of data. If None, it will start at the day when data is available
        end_data : None or a datetime.date object, optional, default None, the end date of data. If None, the day this call is made will be used
        level : 'state' or 'county', optional, default 'state', the geographic resolution of data.
        count : boolean, optional, default True, use count of proportion in 100000 population
        cumulated : boolean, optional, default False, use cumulated data or number of new cases
        mobility_level : int, optional, default 1, which mobility data is used. 
            1 : 'completely_home_prop',
            2 : 'full_time_work_prop',
            3 : 'part_time_work_prop',
            4 : 'median_home_dwell_time'
            
        Returns:
        ========
        if self.merge is True, return full_data, the inner merge of case_data and mobility_data
        if self.merge is False, return cases_data and mobility_data.
        '''
        
        assert isinstance(mobility_level, int), 'mobility_level should be an integer between 1 and 4'
        assert 1<=mobility_level<=4, 'mobility_level should be an integer between 1 and 4'
        
        case_data = self.query_response(source, signal, start_date, end_date, level, count, cumulated)
        
        mobility_data = self.data_loader.query(data_source='safegraph',
                                               signal=self.safegraph_keys[mobility_level],
                                               start_date=start_date,
                                               forecast_date=end_date,
                                               geo_type=level)
        
        if case_data is not None:
            case_data = case_data[['geo_value', 'time_value', 'value']]
            case_data.rename({'value':'case_value', 'time_value':'date'}, axis=1, inplace=True)
            case_data.reset_index(inplace=True, drop=True)
            case_data['time'] = case_data['date'].apply(lambda x: (x.date() - BASE_DATE).days)
            case_data['dayofweek'] = case_data['date'].apply(lambda x : x.dayofweek)
        if mobility_data is not None:
            mobility_data = mobility_data[['geo_value', 'time_value', 'value']]
            mobility_data.rename({'value':'mobility_value', 'time_value':'date'}, axis=1, inplace=True)
            mobility_data.reset_index(inplace=True, drop=True)
            mobility_data['time'] = mobility_data['date'].apply(lambda x: (x.date() - BASE_DATE).days)
            
        
        if not self.merge:
            return case_data, mobility_data
        else:
            if mobility_data is None:
                full_data = None
                return full_data
            if case_data is not None:
                full_data = case_data.merge(mobility_data, on=['geo_value', 'date', 'time'], how='inner')
                return full_data
    
    @staticmethod
    def average_pooling(input, period, end_date):
        '''
        average pooling of case data and mobility data in `input` for every `period` days. 
        If the sample size is not a multiple of `period`, the first few days will be dropped
        
        Params:
        =======
        period : int, the length of each period for an average pooling
        input : pandas.DataFrame object
        '''
        
        assert isinstance(input, pd.DataFrame), '`input` must be a pandas.DataFrame object'
        assert 'geo_value' in input.columns, 'there should be a column named \'geo_value\' in input'
        assert isinstance(period, int) and period>0, '`period` should be a positive integer'
        assert isinstance(end_date, datetime.date), '`end_date` should be a datetime.date object'
        assert end_date > BASE_DATE, '`end_date` should be after Mar 1st, 2020'
        
        #filter data frame
        columns = [name for name in ['case_value', 'mobility_value'] if name in input.columns]
        new_columns = ['geo_value', 'time'] + columns
        input = input[new_columns]
        input = input.reset_index(inplace=False, drop=True)
        max_time = (end_date - BASE_DATE).days
        min_time = min(input['time'])
        N = (max_time - min_time + 1) // period
        min_time = max_time - N * period + 1
        input = input[input['time'].apply(lambda x : min_time <= x <= max_time)]
        print(max_time, min_time, N)
        
        #collect existing date
        recorder = {}
        for i in range(input.shape[0]):
            geo = input.loc[i, 'geo_value']
            time = input.loc[i, 'time']
            record = recorder.get(geo, None)
            if record is None or record['time'] < time:
                # new latest record, update existing record
                recorder[geo] = {'time':time}
                for name in columns:
                    recorder[geo][name] = input.loc[i, name] if name=='mobility_value' else np.nan
        
        #create impute data
        imputed_data = {'time':[], 'geo_value':[]}
        for name in columns:
            imputed_data[name] = []
        for geo in recorder.keys():
            record = recorder[geo]
            for time in range(record['time']+1, max_time+1):
                imputed_data['time'].append(time)
                imputed_data['geo_value'].append(geo)
                for name in columns:
                    imputed_data[name].append(record[name])
        imputed_data = pd.DataFrame(imputed_data)
        input = input.append(imputed_data, ignore_index=True)
        
        pooled = {}
        for geo in recorder.keys():
            pooled[geo] = {}
            for num in range(N):
                pooled[geo][num] = {}
                for name in columns:
                    pooled[geo][num][name] = []
        
        for i in range(input.shape[0]):
            geo = input.loc[i, 'geo_value']
            time = input.loc[i, 'time']
            num = (time - min_time) // period
            for name in columns:
                pooled[geo][num][name].append(input.loc[i, name])
                
        result = dict(zip(new_columns, [[] for _ in new_columns]))
        for geo in recorder.keys():
            for num in range(N):
                result['geo_value'].append(geo)
                result['time'].append(num)
                for name in columns:
                    values = pooled[geo][num][name]
                    result[name].append(np.nanmean(values))
        result = pd.DataFrame(result)
            
        return result
        
                    
        
        
        