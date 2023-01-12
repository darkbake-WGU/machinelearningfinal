#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from outlier_cleaner import outlierCleaner
import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'from_this_person_to_poi', 'bonus', 'exercised_stock_options'] # You will need to use more features

### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "rb") as data_file:
#    data_dict = pickle.load(data_file)
    
data_dict = pickle.load(open("final_project_dataset.pkl", "rb") ) 
###creating dataFrame from dictionary - pandas 
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float) 
print( df.describe())
    

### Task 2: Remove outliers

cleaned_data = outlierCleaner(df)


### Task 3: Create new feature(s)

### Create new feature: bonus_to_salary ratio
cleaned_data['bonus_to_salary'] = cleaned_data['bonus'] / cleaned_data['salary']

# Verify that the new feature was added correctly
print(cleaned_data.describe().loc[:,['bonus_to_salary']])

print(cleaned_data.describe())

print(f'shape of cleaned_data: {cleaned_data.shape}')

### Task 3B: Apply scaling to the features

#Import files necessary for scaling
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the features using fit_transform
# The fit_transform method applies the scaler to the features and returns the scaled features
scaled_features = scaler.fit_transform(cleaned_data[features_list[1:]])
scaled_features_df = pd.DataFrame(scaled_features,columns=cleaned_data[features_list[1:]].columns)

print(scaled_features_df.describe())

### Store to my_dataset for easy export below.
my_dataset = scaled_features_df

print(f'shape of my_dataset: {my_dataset.shape}')
print(f'shape of feature_list: {len(features_list)}')

#features_list2 = ['salary', 'from_this_person_to_poi', 'bonus', 'exercised_stock_options']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()






















### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)