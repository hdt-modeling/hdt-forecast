import pandas as pd
import numpy as np
import datetime
import scipy
import math
import tqdm
import tensorflow as tf

from . import metrics
from . import data_feeder
from forecasters import ArmadilloV1, ARLIC

from hdt_util.delay import Delay
from hdt_util.conv1d import * 
from hdt_util.weekday import Weekday, dow_adjust_cases

import warnings
warnings.filterwarnings("ignore")

CURRENT_DELAY = 4

DEFAULT_DATA_SOURCE = {'source':'jhu-csse', 
                       'signal':'deaths',
                       'count':True, 
                       'cumulated':False,
                       'mobility_level':1,
                       'li_source':'fb-survey',
                       'li_signal':'smoothed_cli',
                       'case_source':'indicator-combination',
                       'case_signal':'confirmed_7dav_cumulative_num'}

class evaluator:
    
    def __init__(self, cache_loc):
        '''
        Initialize evaluator class with location of request cache, so that all evaluation for all models can use the same cache_loc
        '''
        assert isinstance(cache_loc, str), '`cache_loc` must be a string'
        
        self.cache_loc = cache_loc
        self.relations = {'MAE':self.MAE}
        self.delay = CURRENT_DELAY
        self._update_source(DEFAULT_DATA_SOURCE, True)
    
    def update_parameters(self, start_date, end_date, max_prediction_length=1, period=7, min_train=10, method='mean', delay=CURRENT_DELAY):
        '''
        Update dates info for evaluation, also check if dates are valid
        
        Params:
        =======
        start_date : datetime.date object, the start time for training data
        end_date : datetime.date object, the end time for training data
        max_prediction_length : int, positive, at most how many periods to predict
        min_train : int, positive, how many periods required for training
        method : string or a function, how the pooling for each period should be done. If a string, should be one of 'mean', 'max' and 'min'. Case insensitive. Default 'mean'. If a function, it has to return a scalar with a list of scalar (might include np.nan) as input
        '''
        self.delay = delay
        
        assert isinstance(start_date, datetime.date), '`start_date` has to be a `datetime.date` object'
        assert isinstance(end_date, datetime.date), '`end_date` has to be a `datetime.date` object'
        assert end_date < datetime.date.today(), '`end_date` cannot be later than today'
        assert start_date < end_date, '`start_date` has to be prior to `end_date`'
        
        assert isinstance(max_prediction_length, int) and max_prediction_length>0, '`max_prediction_length` has to be a positive integer'
        assert isinstance(period, int) and period > 0, '`period` has to be a positive integer'
        assert isinstance(min_train, int) and min_train > 0, '`min_train` has to be a positive integer'
        assert math.ceil(((end_date - start_date).days - self.delay) / period) >= min_train, 'not enough training data for fitting model'
        
        if isinstance(method, str):
            method = method.lower()
        if method not in ['mean', 'max', 'min']:
            method = 'mean'
        
        self.start_date = start_date
        self.end_date = end_date
        self.max_prediction_length = max_prediction_length
        self.period = period
        self.min_train = min_train
        self.method = method
        
    def evaluate_model(self, *args, **kwargs):
        raise NotImplemented
    
    @staticmethod
    def MAE(y_true, y_pred):
        return metrics.MAE(y_true=y_true, y_pred=y_pred)
    
    def _update_source(self, source, init=False):
        if init:
            self.data_source = {}
        for key, value in source.items():
            self.data_source[key] = value
    
class ArmadilloV1_evaluator(evaluator):
    
    def __init__(self, cache_loc, start_date, end_date, max_prediction_length=1, period=7, min_train=10, method='mean', delay=CURRENT_DELAY):
        
        super(ArmadilloV1_evaluator, self).__init__(cache_loc)
        self.update_parameters(start_date, end_date, max_prediction_length, period, min_train, method, delay)
    
    def evaluate_model(self, model_args, geo_type='state', geo_values=None, data_source_args=None, metrics=[]):
        
        num_days = (self.end_date - self.start_date).days + 1 # how many days in total in training data
        num_period = math.floor(num_days / self.period) # how many periods in the training data
        real_prediction_dates = [self.end_date + datetime.timedelta(days=self.period * i) for i in range(1, self.max_prediction_length+1)] # dates that we have to make a prediction
        real_as_of_dates = [date + datetime.timedelta(days=self.delay) for date in real_prediction_dates] # considering delay, what are the as_of_date we need to evaluation each prediction
        today = datetime.date.today()
        if real_as_of_dates[0] > today:
            print('No enough data to evaluate the model! You have to wait until {} for evaluation data to be available'.format(real_as_of_dates[0]))
            return None, None
        else:
            real_as_of_dates = [date for date in real_as_of_dates if date <=today]
        
        real_prediction_dates = [self.end_date] + real_prediction_dates # although self.end_date is included here, there would be no prediction of it
        
        model = ArmadilloV1(**model_args)
        loader = data_feeder.ArmadilloV1_feeder(self.cache_loc)
        
        if data_source_args is not None:
            self._update_source(data_source_args)
        
        source = self.data_source['source']
        signal = self.data_source['signal']
        count = self.data_source['count']
        cumulated = self.data_source['cumulated']
        mobility_level = self.data_source['mobility_level']
        
        print('loading_data')
        train_data = loader.get_data(source=source, 
                                     signal=signal, 
                                     start_date=self.start_date,
                                     end_date=self.end_date, 
                                     level=geo_type, 
                                     count=count, 
                                     cumulated=cumulated,
                                     mobility_level=mobility_level)
        print('data loaded')
        if geo_values is not None:
            train_data = loader.area_filter(train_data, geo_values)
        train_data['case_value'] = train_data['case_value'].apply(lambda x : max(0, x))
        
        geo_value_candidates = train_data['geo_value'].unique()
        
        geo_values = []
        prediction_length = []
        real_value = []
        predicted_value = []
        
        for geo_value in tqdm.tqdm(geo_value_candidates):
            
            temp_train = train_data[train_data['geo_value'] == geo_value]
            temp_train = temp_train.sort_values(by='time')
            avg_temp_train = loader.pooling(input=temp_train, period=self.period, end_date=self.end_date, method=self.method)
            avg_temp_train.dropna(how='any', inplace=True) # the last entry may have nan for case numbers
            effective_prediction_length = len(real_as_of_dates)
            
            DC = avg_temp_train['time'].values
            DC = scipy.stats.gamma.pdf(DC*self.period, scale=model.args['gamma_scale'], a=model.args['gamma_shape'])
            DC = DC/np.sum(DC) * model.args['death_rate']
            
            args = {'M':avg_temp_train['mobility_value'].values,
                    'DC':DC,
                    'y_true':avg_temp_train['case_value'].values}
            
            model.fit(args)
            
            pred = model.forecast(l = int(num_period - 1 - avg_temp_train['time'].values[-1]) + effective_prediction_length) #'time' starts with 0, that's why we use `num_period - 1` here, as avg_temp_train.values[-1] is an index
            pred = pred[-effective_prediction_length:]
            
            for i, value in enumerate(pred):
                temp = loader.get_data(source=source, 
                                       signal=signal, 
                                       start_date=real_prediction_dates[i],
                                       end_date=real_as_of_dates[i], 
                                       level=geo_type, 
                                       count=count, 
                                       cumulated=cumulated,
                                       mobility_level=mobility_level)
                if temp is None:
                    print('start_data {}, end_date {}, result is None'.format(real_prediction_dates[i], real_as_of_dates[i]))
                else:
                    geo_values.append(geo_value)
                    prediction_length.append(i+1)
                    predicted_value.append(value)
                    temp = temp[temp['geo_value']==geo_value]
                    temp = loader.pooling(temp, self.period, real_prediction_dates[i+1], method=self.method)
                    real_value.append(temp['case_value'].values[0])
                    
        prediction_result = pd.DataFrame({'geo_value':geo_values,
                                          'prediction_length':prediction_length,
                                          'real_value':real_value,
                                          'predicted_value':predicted_value})
        
        evaluation_result = {}
        geo_values = prediction_result['geo_value'].unique()
        for metric, params in metrics:
            metric = metric.upper()
            if metric in self.relations.keys():
                func = self.relations[metric]
                errors = []
                for area in geo_values:
                    y_true = prediction_result[prediction_result['geo_value']==area]['real_value'].values
                    y_pred = prediction_result[prediction_result['geo_value']==area]['predicted_value'].values
                    if params is not None:
                        errors.append(func(y_true, y_pred, **params))
                    else:
                        errors.append(func(y_true, y_pred))
                evaluation_result[metric] = errors
        
        evaluation_result[geo_value] = geo_values
        evaluation_result = pd.DataFrame(evaluation_result)
                
        return prediction_result, evaluation_result
    
class ARLIC_evaluator(evaluator):
    
    def __init__(self, cache_loc, start_date, end_date, max_prediction_length=1, period=7, min_train=30, method='mean', delay=CURRENT_DELAY):
        
        super(ARLIC_evaluator, self).__init__(cache_loc)
        self.update_parameters(start_date, end_date, max_prediction_length, period, min_train, method, delay)
    
    def evaluate_model(self, model_args, geo_type='state', geo_values=None, data_source_args=None, metrics=[]):
        
        delay_dist = model_args['delay_dist']
        model = ARLIC(**model_args)
        optimizer = tf.keras.optimizers.Adam(lr=0.1)
        loss = tf.keras.losses.MSE
        
        def scheduler(epoch, lr):
            LR = [0.1, 0.04, 0.01, 0.004, 0.001, 0.0004]
            if epoch < 50000:
                return LR[0]
            elif epoch < 60000:
                return LR[1]
            elif epoch < 70000:
                return LR[2]
            elif epoch < 80000:
                return LR[3]
            elif epoch < 90000:
                return LR[4]
            else:
                return LR[5]
        
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(
            scheduler)
        
        model_path = "ARLIC_MODEL_PATH"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            verbose=0,
            monitor="mean_squared_error",
            save_best_only=True,
            save_weights_only=True
        )
        
        callbacks = [
            checkpoint_callback,
            scheduler_callback,
        ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss, 
            metrics=[loss],
        )

        loader = data_feeder.ARLIC_feeder(self.cache_loc)
        
        if data_source_args is not None:
            self._update_source(data_source_args)
        
        li_source = self.data_source['li_source']
        li_signal = self.data_source['li_signal']
        case_source = self.data_source['case_source']
        case_signal = self.data_source['case_signal']
        
        print('loading_data')
        train_data = loader.get_data(case_source=case_source,
                                     case_signal=case_signal,
                                     li_source=li_source, 
                                     li_signal=li_signal, 
                                     start_date=self.start_date,
                                     end_date=self.end_date, 
                                     level=geo_type)
        print('data loaded')
        
        if geo_values is not None:
            train_data = loader.area_filter(train_data, geo_values)
        
        geo_value_candidates = train_data['geo_value'].unique()
        
        geo_values = []
        prediction_length = []
        real_value = []
        predicted_value = []
        
        for geo_value in tqdm.tqdm(geo_value_candidates):
            
            #prepara data
            temp_data = train_data[train_data['geo_value'] == geo_value]
            temp_last_day = temp_data['time'].max()
            temp_first_day = temp_data['time'].min()
            day_count_1 = temp_last_day - temp_first_day + 1
            day_count_2 = self.min_train + self.max_prediction_length
            if day_count_1 < day_count_2:
                print('No enough data for {}. Requires at least {} days. Have {} days'.format(geo_value, day_count_2, day_count_1))
                continue
                
            temp_data = temp_data.sort_values(by='time')
            temp_train = temp_data[temp_data['time'] <= temp_last_day - self.max_prediction_length]
            temp_eval = temp_data[temp_data['time'] > temp_last_day - self.max_prediction_length]
            
            temp_cases = temp_train['case_value'].values
            temp_cases = tf.reshape(temp_cases, shape=(1,-1,1))
            
            temp_li = temp_train['li_value'].values
            cases_mean = tf.reduce_mean(temp_cases)
            cases_std = tf.math.reduce_std(temp_cases)
            
            #normalize li and deconvolve
            li_mean = tf.reduce_mean(temp_li)
            li_std = tf.math.reduce_std(temp_li)
            li = pd.DataFrame({'value':temp_li, 'time_value':temp_train['date'].values})
            li.value = tf.clip_by_value((cases_std*(li.value-li_mean)/li_std) + cases_mean, clip_value_min=0,clip_value_max=float('inf'))       

            temp_li = Delay.deconv(li, delay_dist)
            temp_li = tf.reshape(temp_li, shape=(1,-1,1))
            
            #fit model and reload best weights
            args = {
                "x":temp_li,
                "y":temp_cases,
                "epochs":1000,
                "verbose":0,
                "callbacks":callbacks
            } 
            model.fit(args)
            model.load_weights(model_path).expect_partial()
            
            #prediction
            temp_li = tf.pad(temp_li, paddings=[[0,0],[model.p-1,0],[0,0]])
            temp_It = model(temp_li)
            temp_It = tf.clip_by_value(temp_It, clip_value_min=0, clip_value_max=float('inf'))
            temp_It = tf.pad(temp_It, paddings=[[0,0],[model.delay_dist.shape[0]-1,0],[0,0]]) 
            
            n = self.max_prediction_length # predict until end_day
            pred = model.forecast(temp_li, n)
            pred = tf.reshape(pred, shape=(1,-1,1))
            temp_It = tf.concat([temp_It, pred], axis=1)
            temp_It = tf.clip_by_value(temp_It, clip_value_min=0, clip_value_max=float('inf'))
            
            pred = Delay.conv(temp_It, delay_dist)
            
            pred = tf.reshape(pred, shape=-1)[:-1].numpy() 
            pred = pred[-self.max_prediction_length:] # only keep the values from end_date to end_date+self.max_prediction_length
            
            days_available = temp_eval['time'].values - temp_last_day - 1 + self.max_prediction_length
            
            for i, day in enumerate(days_available):
                geo_values.append(geo_value)
                prediction_length.append(day+1)
                predicted_value.append(pred[day])
                real_value.append(temp_eval['case_value'].values[i])
                    
        prediction_result = pd.DataFrame({'geo_value':geo_values,
                                          'prediction_length':prediction_length,
                                          'real_value':real_value,
                                          'predicted_value':predicted_value})
        
        evaluation_result = {}
        geo_values = prediction_result['geo_value'].unique()
        for metric, params in metrics:
            metric = metric.upper()
            if metric in self.relations.keys():
                func = self.relations[metric]
                errors = []
                for area in geo_values:
                    y_true = prediction_result[prediction_result['geo_value']==area]['real_value'].values
                    y_pred = prediction_result[prediction_result['geo_value']==area]['predicted_value'].values
                    if params is not None:
                        errors.append(func(y_true, y_pred, **params))
                    else:
                        errors.append(func(y_true, y_pred))
                evaluation_result[metric] = errors
        
        evaluation_result[geo_value] = geo_values
        evaluation_result = pd.DataFrame(evaluation_result)
                
        return prediction_result, evaluation_result
                