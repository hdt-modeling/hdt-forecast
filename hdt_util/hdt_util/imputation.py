"""
This file contails different default imputation methods for data, for both confirmed/death counts and mobility values
"""

import datetime
import scipy
import pandas as pd

from collections import defaultdict

class imputation:
    '''
    Imputation class with only static methods
    '''
    
    @staticmethod
    def impute_with_scipy(df, method):
        '''
        Params:
        =======
        Impute missing values with scipy.interpolate module. For each value column, if there were missing values that can be imputed with interpolation, it will be imputed. Missing values that can only be imputed by extrapolation will remain missing.
        
        df: pandas.DataFrame containing all data. The dataframe should have 
            (1) a column called 'date', with datetime.dates objects, the dates for data
            (2) a column called 'time', indicating how many days the date is after BASE_DATE (now 2020-02-29)
            (3) a column called 'geo_value', indication the location of this data
            (3) all other columns with data to be imputed
            
        method: string, the method used to do interpolation. Now supporting:
            'linear': linear imputation
            'bspline': fit B-Spline
            'zero': fit a zeroth order spline
            'slinear': fit a first order spline
            'quadratic': fit a second order spline
            'cubic': fit a third order spline
        '''
        
        assert isinstance(df, pd.DataFrame), '`df` should be a pandas.DataFrame object'
        
        na_counts = df.isna().sum(axis=0)
        if na_counts.values.max() == 0: # no missing value in the whole data frame
            return df
        
        columns = df.columns
        assert 'date' in columns, '`df` should have a column with dates, named `date`'
        assert 'geo_value' in columns, '`df` should have a column with dates, named `geo`'
        assert 'time' in columns, '`df` should have a column with relative dates (how many days after 2020-02-09), named `time`'
        
        locations = df['geo_value'].unique()
        new_df = pd.DataFrame({})
        
        for geo in locations:
            
            temp_df = df[df['geo_value'] == geo]
            temp_df.reset_index(drop=True, inplace=True)
            na_counts = temp_df.isna().sum(axis=0)
            if na_counts.values.max() == 0:#no missing value for this geo
                new_df = new_df.append(temp_df)
                continue
            
            temp_start_time = temp_df.time.min()
            for column in columns:
                if na_counts[column] == 0:#this column has no missing value
                    continue
                    
                temp_sub_df = temp_df[temp_df[column].notna()]
                existing_time = temp_sub_df.time.values
                existing_values = temp_sub_df[column].values
                possible_time = [time for time in range(existing_time.min(), existing_time.max()+1)]
                imputed = imputation._impute_with_scipy(existing_time, existing_values, possible_time, method)
                assert len(imputed) == len(possible_time)
                
                #imputation is finished. Now use the imputed value to cover know NAs
                for ind, time in enumerate(possible_time):
                    temp_df.loc[time-temp_start_time, column] = imputed[ind]
            
            new_df = new_df.append(temp_df)
        
        new_df.reset_index(drop=True, inplace=True)
        return new_df

    @staticmethod
    def _impute_with_scipy(existing_x, existing_y, possible_x, method):
        
        if method == 'bspline':
            t, c, k = scipy.interpolate.splrep(existing_x, existing_y)
            function = scipy.interpolate.BSpline(t, c, k)
        else:
            function = scipy.interpolate.interp1d(existing_x, existing_y, kind=method)
        
        imputed = function(possible_x)
        
        return imputed