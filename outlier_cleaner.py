#!/usr/bin/python
from scipy import stats
import numpy as np
import pandas as pd
from typing import Dict

#This function takes the dictionary as a data variable.
#The threshold is the z-score you would like to cap.
def outlierCleaner(data: pd.DataFrame, threshold: float = 3) :
    """
        A function that takes a dictionary and returns a similar dictionary with outliers capped using the Z-score method and any outliers capped are printed out:
    """
    #This function makes a copy of the dataframe as cleaned_data
    cleaned_data = data.copy()
    
    #It skips non-numeric column poi
    for feature in data.columns:
        if feature in ['poi']:
            continue
        
        #It finds the mean and std of that column and computes the z-scores of the values in it
        mean = data[feature].mean()
        std = data[feature].std()
        z_scores = (data[feature] - mean) / std
        
        #This would be the score of a value with a standard deviation of 3, which is our cap
        cap = 3*std + mean
        
        #The function uses the z_scores object to determine when to cap 
        cleaned_data[feature] = np.where(np.abs(z_scores) > threshold, cap, data[feature])
        outliers = data[np.abs(z_scores) > threshold]
        #If there are outliers, the function prints them to the console!
        if not outliers.empty:
                print(f"Outliers capped for feature {feature} for person {outliers.index}")
    return cleaned_data

def replace_nan_with_mean(dataframe):
#This function replaces the nan values with the mean of the values in that column
    print('Function activated')
    #Iterate through the columns in the dataframe
    for column in dataframe.columns:
        #Skip poi column
        if column in ['poi']:
            continue
        print('NAN values replaced')
        #Calculate the mean of the column for replacement
        mean = dataframe[column].mean()
        
        #Fill null values with the mean
        dataframe[column].fillna(mean, inplace=True)
    
    #Return a dataframe
    return dataframe
