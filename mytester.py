# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 06:39:17 2023

@author: zebov
"""
import sys
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tester import test_classifier
from tester import dump_classifier_and_data
from outlier_cleaner import replace_nan_with_mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data_dict = pickle.load(open("final_project_dataset.pkl", "rb") ) 

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees', 'bonus_to_salary']
features_list2 = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

print("Removing outliers")
###Removing an entry with an entirely empty data set
data_dict.pop("LOCKHART EUGENE E")

#Removing the TOTAL person as it is not a person at all
data_dict.pop("TOTAL")

print('Replacing NAN with mean values')
#Additional functionality was added to the outlier_cleaner.py
#This function replaces NAN values with the column mean
data_dict = replace_nan_with_mean(data_dict, features_list2)

### Task 3: Create new feature(s)

for key in data_dict:
    data_dict[key]['bonus_to_salary'] = (data_dict[key]['bonus'] / (data_dict[key]['salary'] + 0.01))


# Create a dictionary to store the values for each feature
feature_values = {feature: [] for feature in data_dict[key]}

# Add the values for each feature for each person to the dictionary
for person in data_dict:
    for feature in data_dict[person]:
        try:
            float_val = float(data_dict[person][feature])
            feature_values[feature] = np.append(feature_values[feature], float_val)
        except ValueError:
            print(f'{data_dict[person][feature]} is not a numeric value')


data_dict[key].pop('email_address')

financial_features = ['salary', 'bonus', 'exercised_stock_options', 'restricted_stock', 'total_payments', 'expenses','total_stock_value', 'deferred_income', 'long_term_incentive']
behavioral_features = ['to_messages','from_messages','from_poi_to_this_person','from_this_person_to_poi','other']

#Convert to data frame 
## Exploring the dataset through pandas.Dataframe
data_frame = pd.DataFrame.from_dict(data_dict, orient='index')
data_frame.head()

#Doing PCA analysis on financial features and converting them to one feature
pca = PCA(n_components=1)
pca.fit(data_frame[financial_features])
pcaComponents = pca.fit_transform(data_frame[financial_features])
data_frame['financial'] = pcaComponents

#Setting up the strategic features, which use behavioral and the one financial feature
strategic_features = behavioral_features + ['financial']

print(data_frame['financial'])
## Converting back the pandas Dataframe to the dictionary structure
my_dataset = data_frame.to_dict(orient='index')

#Using strategic features
#features_list = strategic_features
features_list = ['poi'] + financial_features + ['financial'] + behavioral_features

### Import the necessary files
from sklearn.ensemble import AdaBoostClassifier

### Import the RandomForestClassifier from sklearn.ensemble
###Import the pipeline and scaler as well
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

### Import GridSearchCV
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Import the required functions from sklearn.metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Here we are splitting the data into testing and training segments
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

cv = StratifiedShuffleSplit(n_splits=1000, random_state = 42)

#Create the pipeline
pipe = Pipeline([
    #('scaler', StandardScaler()),
    ('selector', SelectKBest()),
    ('classifier', AdaBoostClassifier())
])

### Define the parameter grid for the AdaBoostClassifier
param_grid = {'classifier__n_estimators': [10, 25, 50, 100],
              'classifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
              'classifier__algorithm': ['SAMME', 'SAMME.R'],
              'selector__k': [2,4,6,8,10]}

print("Create grid search")
### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='recall')

print("Fit the grid search")
### Fit the grid search object to the data
grid_search.fit(features, labels)

print("AdaBoost:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

### Make predictions using the best estimator
best_estimator = grid_search.best_estimator_
pred = best_estimator.predict(features)

### Print the accuracy, precision, recall, and f1-score
print("Accuracy: ", accuracy_score(labels, pred))
print("Precision: ", precision_score(labels, pred))
print("Recall: ", recall_score(labels, pred))
print("F1-score: ", f1_score(labels, pred))



pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1, n_estimators=10))
])

#clf = RandomForestClassifier(criterion='entropy', max_depth=5,min_samples_leaf=1,min_samples_split=2,n_estimators=10)

###Here, we use the official test_classifier function to check our results.
test_classifier(pipe, my_dataset, features_list)

dump_classifier_and_data(pipe, my_dataset, features_list)