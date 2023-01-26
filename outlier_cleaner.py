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

def replace_nan_with_mean(data_dict, features_list):
    #Cycle through the features
    for feature in features_list:
        #Get the mean of the feature
        mean = np.mean([data_dict[key][feature] for key in data_dict if data_dict[key][feature] != 'NaN'])
        #Cycle through the keys in the dictionary and replace 'NaN' with mean
        for key in data_dict:
            if data_dict[key][feature] == 'NaN':
                data_dict[key][feature] = mean
    return data_dict

def feature_nulls_analyze(data_dict, features_list):
    import matplotlib.pyplot as plt
    # Create a new dictionary to store the counts of NaN values for each feature
    nan_counts = {feature: 0 for feature in features_list}
    # Iterate over the data dictionary
    for key in data_dict:
        for feature in features_list:
            if data_dict[key][feature] == 'NaN':
                nan_counts[feature] += 1
    # Calculate the percentage of NaN values for each feature
    nan_counts = {feature: count / len(data_dict) * 100 for feature, count in nan_counts.items()}

    # Create a bar chart to visualize the results
    plt.bar(nan_counts.keys(), nan_counts.values())

    # Add labels and title
    plt.xlabel('Features')
    plt.ylabel('Percentage of NaN values')
    plt.title('Percentage of NaN values by feature')
    
    #Rotate the x labels
    plt.xticks(rotation=90)

    # Display the chart
    plt.show()
    
def count_nan_entries(data_dict):
    nan_count = {}
    for key, value in data_dict.items():
        nan_count[key] = sum(map(lambda x: x=='NaN', value.values()))
    return nan_count