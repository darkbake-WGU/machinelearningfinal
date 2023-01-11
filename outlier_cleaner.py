#!/usr/bin/python
from scipy import stats
import numpy as np

def outlierCleaner(data_dict):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    ### your code goes here

    for col in data_dict.columns:
        print("working with ",col)
        if (((data_dict[col].dtype)=='float64') | ((data_dict[col].dtype)=='int64')):
            percentiles = data_dict[col].quantile([0.01,0.99]).values
            data_dict[col][data_dict[col] <= percentiles[0]] = percentiles[0]
            data_dict[col][data_dict[col] >= percentiles[1]] = percentiles[1]
        else:
            data_dict[col]=data_dict[col]
    return data_dict

