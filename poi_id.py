#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from outlier_cleaner import outlierCleaner
from outlier_cleaner import replace_nan_with_mean
from outlier_cleaner import feature_nulls_analyze
import pandas as pd
import numpy as np
import numbers
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### We will use every single numeric feature available
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] # You will need to use more features
features_list2 = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] # You will need to use more features
### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "rb") as data_file:

### We are also going to convert the data set into a pandas dataframe for use
### in the outlierCleaner() function that I rewrote.
    
data_dict = pickle.load(open("final_project_dataset.pkl", "rb") ) 
data_frame = pd.DataFrame.from_dict(data_dict, orient='index')

# Iterate over the dictionary and replace 'NaN' strings with np.nan
#for key in data_dict:
#    for feature in features_list:
#        if data_dict[key][feature] == 'NaN':
#            data_dict[key][feature] = np.nan

feature_nulls_analyze(data_dict, features_list)
#Select only the features in features_list

print('Replacing NAN with mean values')
#Additional functionality was added to the outlier_cleaner.py
#This function replaces NAN values with the column mean
selected_features = replace_nan_with_mean(data_dict, features_list2)

### Describe the data to check that everything is going okay
print("Check the data frame yet again")
print(dict(list(data_dict.items())[:5]))

### Task 2: Remove outliers
### This outlierCleaner() function has been rewritten
#Run outlier cleaner
print("Cleaning Outliers has been cancelled")

### Task 3: Create new feature(s)

for key in data_dict:
    data_dict[key]['bonus_to_salary'] = (data_dict[key]['bonus'] / (data_dict[key]['salary'] + 0.01))

# Verify that the new feature was added correctly
print("Verifying that bonus_to_salary was added correctly")
for key in data_dict:
    print(data_dict[key]['bonus_to_salary'])


### Store to my_dataset for easy export below.
my_dataset = data_dict

print("Printing cleaned data")
print(my_dataset)


### Import necessary files
from sklearn.feature_selection import SelectKBest, f_classif

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Create an instance of the SelectKBest object
selector = SelectKBest(f_classif, k=19)

### Fit the SelectKBest object to the features and labels
selector.fit(features, labels)

### Get the scores for each feature
scores = selector.scores_

### Create a dictionary of features and their scores
features_scores = dict(zip(features_list[1:], scores))

### Print the scores of each feature
print("Feature Importance Scores:")
for feature, score in features_scores.items():
    print("{}: {}".format(feature, score))


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### We will be skipping directly to supervised classifiers and trying out
### RandomForest, SVC and AdaBoost. We will be using a grid search method
### in order to determine optimal parameters for each classifier.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### We are going to keep the top 4 features  
features_list = ['poi', 'from_poi_to_this_person', 'shared_receipt_with_poi','director_fees', 'loan_advances']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Here we are splitting the data into testing and training segments
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    

### Import the RandomForestClassifier from sklearn.ensemble
###Import the pipeline and scaler as well
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

### We are going to be using a grid search method in order to determine
### the best parameters for the RandomForest classifier.

### Import GridSearchCV
from sklearn.model_selection import GridSearchCV

### Define the parameter grid for the grid search
param_grid = {'classifier__n_estimators': [10, 50, 100],
              'classifier__max_depth': [None, 5, 10]}

#Create the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])



### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')

### Fit the grid search object to the data
grid_search.fit(features_train, labels_train)

print("RandomForest:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
    
best_estimator = grid_search.best_estimator_
best_estimator.fit(features_train, labels_train)
pred = best_estimator.predict(features_test)

### Import the required functions from sklearn.metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Compute accuracy
accuracy = accuracy_score(labels_test, pred)
print("Accuracy: ", accuracy)

### Compute precision
precision = precision_score(labels_test, pred)
print("Precision: ", precision)

### Compute recall
recall = recall_score(labels_test, pred)
print("Recall: ", recall)

### Compute F1-score
f1 = f1_score(labels_test, pred)
print("F1-score: ", f1)

### Now we are trying SVC

### Import required package
from sklearn.svm import SVC

# Define the parameter grid for the SVC
param_grid = {'classifier__C': [0.1, 1, 10, 100],
              'classifier__kernel': ['linear', 'rbf']}

#
print("Create classifier")
#Create the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])


print("Create grid search")
### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')

print("Fit the grid search")
### Fit the grid search object to the data
grid_search.fit(features_train, labels_train)

print("SVC:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
#

best_estimator = grid_search.best_estimator_
best_estimator.fit(features_train, labels_train)
pred = best_estimator.predict(features_test)


### Compute accuracy
accuracy = accuracy_score(labels_test, pred)
print("Accuracy: ", accuracy)

### Compute precision
precision = precision_score(labels_test, pred)
print("Precision: ", precision)

### Compute recall
recall = recall_score(labels_test, pred)
print("Recall: ", recall)

### Compute F1-score
f1 = f1_score(labels_test, pred)
print("F1-score: ", f1)

### Now we will try AdaBoost

### Import the necessary files
from sklearn.ensemble import AdaBoostClassifier

#Create the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', AdaBoostClassifier())
])

### Define the parameter grid for the AdaBoostClassifier
param_grid = {'classifier__n_estimators':[50, 100, 200],
              'classifier__learning_rate':[0.1, 0.5, 1.0]}

print("Create grid search")
### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')

print("Fit the grid search")
### Fit the grid search object to the data
grid_search.fit(features_train, labels_train)

print("AdaBoost:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

### Make predictions using the best estimator
best_estimator = grid_search.best_estimator_
best_estimator.fit(features_train, labels_train)
pred = best_estimator.predict(features_test)

### Print the accuracy, precision, recall, and f1-score
print("Accuracy: ", accuracy_score(labels_test, pred))
print("Precision: ", precision_score(labels_test, pred))
print("Recall: ", recall_score(labels_test, pred))
print("F1-score: ", f1_score(labels_test, pred))

### It looks like we have a winner! We will use the RandomForest classifier
### Write the classifier here with its optimal parameters:
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', AdaBoostClassifier(learning_rate=.5,n_estimators=50))
])

test_classifier(best_estimator, my_dataset, features_list)
























### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipe, my_dataset, features_list)