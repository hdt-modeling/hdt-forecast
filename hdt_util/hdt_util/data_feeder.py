'''
In this file, each model will have its data feeder. Class names will be the model's names 
'''

from .get_covidcast import CovidcastGetter
import pandas as pd
import numpy as np
import datetime
from datetime import date

BASE_DATE = date(2020, 3, 1)

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
        if mobility_data is not None:
            mobility_data = mobility_data[['geo_value', 'time_value', 'value']]
            mobility_data.rename({'value':'mobility_value', 'time_value':'date'}, axis=1, inplace=True)
            mobility_data.reset_index(inplace=True, drop=True)
            
        
        if not self.merge:
            return case_data, mobility_data
        else:
            if mobility_data is None:
                full_data = None
                return full_data
            if case_data is not None:
                full_data = case_data.merge(mobility_data, on=['geo_value', 'date'], how='inner')
                return full_data
        