import pandas as pd
import numpy as np
import datetime
import scipy
import math
import tqdm

from . import metrics
from . import data_feeder
from forecasters.lv_mobility import LVMM

import warnings
warnings.filterwarnings("ignore")

CURRENT_DELAY = 4

DEFAULT_DATA_SOURCE = {'source':'jhu-csse', 
                       'signal':'deaths', 
                       'level':'state', 
                       'count':True, 
                       'cumulated':False,
                       'mobility_level':1}

class evaluator:
    
    def __init__(self, cache_loc):
        '''
        Initialize evaluator class with location of request cache, so that all evaluation for all models can use the same cache_loc
        '''
        assert isinstance(cache_loc, str), '`cache_loc` must be a string'
        
        self.cache_loc = cache_loc
        self.relations = {'MAE':self.MAE}
    
    def update_parameters(self, start_date, end_date, max_prediction_length=1, period=7, min_train=10, metrics=[]):
        '''
        Update dates info for evaluation, also check if dates are valid
        
        Params:
        =======
        start_date : datetime.date object, the start time for training data
        end_date : datetime.date object, the end time for training data
        max_prediction_length : int, positive, at most how many periods to predict
        min_train : int, positive, how many periods required for training
        metrics : List<str>, a list of metrics to calculate
        '''
        
        assert isinstance(start_date, datetime.date), '`start_date` has to be a `datetime.date` object'
        assert isinstance(end_date, datetime.date), '`end_date` has to be a `datetime.date` object'
        assert end_date < datetime.date.today(), '`end_date` cannot be later than today'
        assert start_date < end_date, '`start_date` has to be prior to `end_date`'
        
        assert isinstance(max_prediction_length, int) and max_prediction_length>0, '`max_prediction_length` has to be a positive integer'
        assert isinstance(period, int) and period > 0, '`period` has to be a positive integer'
        assert isinstance(min_train, int) and min_train > 0, '`min_train` has to be a positive integer'
        assert math.ceil(((end_date - start_date).days - CURRENT_DELAY) / period) >= min_train, 'not enough training data for fitting model'
        
        assert isinstance(metrics, list) and all(isinstance(name, str) for name in metrics), '`metrics` must be a list of names (as strings) of evaluation metrics'
        
        self.start_date = start_date
        self.end_date = end_date
        self.max_prediction_length = max_prediction_length
        self.period = period
        self.min_train = min_train
        self.metrics = metrics
        
    def evaluate_model(self, *args, **kwargs):
        raise NotImplemented
        
    def MAE(self, y_true, y_pred):
        return metrics.MAE(y_true=y_true, y_pred=y_pred)
    
class Valerie_and_Larry_evaluator(evaluator):
    
    def __init__(self, cache_loc, start_date, end_date, max_prediction_length=1, period=7, min_train=10, metrics=[]):
        
        super(Valerie_and_Larry_evaluator, self).__init__(cache_loc)
        self.update_parameters(start_date, end_date, max_prediction_length, period, min_train, metrics)
    
    def evaluate_model(self, model_args, data_source_args=None):
        
        num_days = (self.end_date - self.start_date).days
        num_period = math.ceil(num_days / self.period)
        real_start_date = self.start_date + datetime.timedelta(days = num_days % self.period)
        real_prediction_dates = [self.end_date + datetime.timedelta(days=self.period * i) for i in range(1, self.max_prediction_length+1)]
        real_as_of_dates = [date + datetime.timedelta(days=CURRENT_DELAY) for date in real_prediction_dates]
        today = datetime.date.today()
        if real_as_of_dates[0] > today:
            print('No enough data to evaluate the model! You have to wait until {} for evaluation data to be available'.format(real_as_of_dates[0]))
            return None, None
        else:
            real_as_of_dates = [date for date in real_as_of_dates if date <=today]
        
        real_prediction_dates = [self.end_date] + real_prediction_dates
        
        model = LVMM(**model_args)
        loader = data_feeder.Valerie_and_Larry_feeder(self.cache_loc)
        
        if data_source_args is None:
            data_source_args = DEFAULT_DATA_SOURCE
        
        source = data_source_args['source']
        signal = data_source_args['signal']
        level = data_source_args['level']
        count = data_source_args['count']
        cumulated = data_source_args['cumulated']
        mobility_level = data_source_args['mobility_level']
        
        print('loading_data')
        train_data = loader.get_data(source=source, 
                                     signal=signal, 
                                     start_date=self.start_date,
                                     end_date=self.end_date, 
                                     level=level, 
                                     count=count, 
                                     cumulated=cumulated,
                                     mobility_level=mobility_level)
        print('data loaded')
        train_data['case_value'] = train_data['case_value'].apply(lambda x : max(0, x))
        
        geo_value_candidates = train_data['geo_value'].unique()
        
        geo_values = []
        prediction_length = []
        real_value = []
        predicted_value = []
        
        for geo_value in tqdm.tqdm(geo_value_candidates):
            
            temp_train = train_data[train_data['geo_value'] == geo_value]
            avg_temp_train = loader.average_pooling(temp_train, self.period, self.end_date)
            avg_temp_train.dropna(how='any', inplace=True)
            effective_prediction_length = len(real_as_of_dates)
            
            DC = avg_temp_train['time'].values
            DC = scipy.stats.gamma.pdf(DC*self.period, scale=model.args['gamma_scale'], a=model.args['gamma_shape'])
            DC = DC/np.sum(DC) * model.args['death_rate']
            
            model.fit(M = avg_temp_train['mobility_value'].values,
                      DC = DC,
                      y_true = avg_temp_train['case_value'].values)
            
            pred = model.forecast(l = int(num_period - 1 - avg_temp_train['time'].values[-1]) + effective_prediction_length) #'time' starts with 0, that's why we use `num_period - 1` here
            pred = pred[-effective_prediction_length:]
            
            for i, value in enumerate(pred):
                geo_values.append(geo_value)
                prediction_length.append(i+1)
                predicted_value.append(value)
                temp = loader.get_data(source=source, 
                                       signal=signal, 
                                       start_date=real_prediction_dates[i],
                                       end_date=real_as_of_dates[i], 
                                       level=level, 
                                       count=count, 
                                       cumulated=cumulated,
                                       mobility_level=mobility_level)
                if temp is None:
                    print('start_data {}, end_date {}, result is None'.format(real_prediction_dates[i], real_as_of_dates[i]))
                temp = temp[temp['geo_value']==geo_value]
                temp = loader.average_pooling(temp, self.period, real_prediction_dates[i+1])
                real_value.append(temp['case_value'].values[0])
                
        return pd.DataFrame({'geo_value':geo_values,
                             'prediction_length':prediction_length,
                             'real_value':real_value,
                             'predicted_value':predicted_value})
                