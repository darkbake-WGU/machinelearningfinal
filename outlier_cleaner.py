#!/usr/bin/python
from scipy import stats
import numpy as np
import pandas as pd

def outlierCleaner(data_dict):
    """
        Clean away the top and bottom 1% of the data
    """

    ### your code goes here

    for col in data_dict.columns:
        print("working with ",col)
        if (((data_dict[col].dtype)=='float64') | ((data_dict[col].dtype)=='int64')):
            percentiles = data_dict[col].quantile([0.01,0.99]).values
            data_dict[col][data_dict[col] <= percentiles[0]] = percentiles[0]
            data_dict[col][data_dict[col] >= percentiles[1]] = percentiles[1]
      
    return pd.DataFrame(data_dict)

