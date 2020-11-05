import pandas as pd
import numpy as np
from datetime import date, timedelta
import tqdm

from . import metrics

import warnings
warnings.filterwarnings("ignore")

CURRENT_DELAY = 4

class evaluator:
    
    def __init__(self, start_date, end_date, prediction_length=1, period=7, min_train=10, metrics=[]):
        '''
        Initialize an evaluator for model evaluation
        
        Params:
        =======
        model : the model class to be evaluated
        model_args : dict, args for the model
        start_date : datetime.date object, the start time for training data
        end_date : datetime.date object, the end time for training data
        prediction_length : int, positive, how many periods to predict
        min_train : int, positive, how many periods required for training
        metrics : List<str>, a list of metrics to calculate
        '''
        
        assert isinstance(start_date, date). '`start_date` has to be a `datetime.date` object'
        assert isinstance(end_date, date). '`end_date` has to be a `datetime.date` object'
        assert start_date < end_date, '`start_date` has to be prior to `end_date`'
        
        assert isinstance(prediction_length, int) and prediction_length>0, '`prediction_length` has to be a positive integer'
        assert isinstance(period, int) and period > 0, '`period` has to be a positive integer'
        assert isinstance(min_train, int) and min_train > 0, '`min_train` has to be a positive integer'
        assert ((end_date - start_date).days - CURRENT_DELAY) // period >= period - 1, 'not enough training data for fitting model'
        
        assert isinstance(metrics, list) and all(isinstance(name, str) for name in metrics), '`metrics` must be a list of names (as strings) of evaluation metrics'
        
        self.start_date = start_date
        self.end_date = end_date
        self.data_source_args = data_source_args
        self.prediciton_end_date = prediciton_end_date
        self.period = self.period
        self.min_train = min_train
        self.metrics = metrics
        
    def evaluate_model(self, model, model_args, data_source_args):
        raise NotImplemented
    
    