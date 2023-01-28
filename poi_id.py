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
from outlier_cleaner import count_nan_entries
from sklearn.model_selection import StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### We will use every single numeric feature available, at least for now.
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] # You will need to use more features
###This was necessary in order to use a certain function later
features_list2 = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] # You will need to use more features
### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "rb") as data_file:
    
data_dict = pickle.load(open("final_project_dataset.pkl", "rb") ) 


###We are going to get an idea of how many nulls exist per feature
feature_nulls_analyze(data_dict, features_list)


###We are going to get an idea of how nany nulls exist per person
nan_counts = count_nan_entries(data_dict)
print("Printing nan counts")
print(nan_counts)

print("Removing outliers")
###Removing an entry with an entirely empty data set
data_dict.pop("LOCKHART EUGENE E")

#Removing the TOTAL person as it is not a person at all
data_dict.pop("TOTAL")



print('Replacing NAN with mean values')
#Additional functionality was added to the outlier_cleaner.py
#This function replaces NAN values with the column mean
data_dict = replace_nan_with_mean(data_dict, features_list2)

### Task 2: Remove outliers
### This outlierCleaner() function has been rewritten
#Run outlier cleaner
print("Automatic outlier cleaner has been cancelled")

### Task 3: Create new feature(s)

for key in data_dict:
    data_dict[key]['bonus_to_salary'] = (data_dict[key]['bonus'] / (data_dict[key]['salary'] + 0.01))

# Verify that the new feature was added correctly
print("Verifying that bonus_to_salary was added correctly")
for key in data_dict:
    print(data_dict[key]['bonus_to_salary'])


### Store to my_dataset for easy export below.
my_dataset = data_dict

print("Printing cleaned data keys")
keys = list(my_dataset["SHARP VICTORIA T"].keys())
print(keys)

#Adding in bonus_to_salary
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees', 'bonus_to_salary'] # You will need to use more features


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

### We are going to keep the features with scores > .5
#features_list = ['poi', 'salary', 'total_payments', 'loan_advances', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock'] # You will need to use more features


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

#Remove feature 4 from the data set, as it is constant

features = np.delete(features, 4, axis=1)
features_test = np.delete(features_test, 4, axis=1)

### Import GridSearchCV
from sklearn.model_selection import GridSearchCV

###Modified the code to take into account the same test as tester.py
cv = StratifiedShuffleSplit(n_splits=100, random_state = 42)

### Define the parameter grid for the grid search

param_grid = {'classifier__n_estimators': [2, 5, 10, 50, 100],
              'classifier__max_depth': [None, 5, 10, 20, 30, 40, 50],
              'classifier__min_samples_split': [2, 5, 10],
              'classifier__min_samples_leaf': [1, 2, 4],
              'classifier__criterion': ['gini', 'entropy'],
              'classifier__max_features': ['auto', 'sqrt']}
              #selector__k': [2,4,6,8,10]}

###Create the pipeline - scaler not necessary for RandomForest
pipe = Pipeline([
    #('scaler', StandardScaler()),
    ('selector', SelectKBest(k=3)),
    ('classifier', RandomForestClassifier())
])


### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='recall')

### Fit the grid search object to the data
grid_search.fit(features, labels)

print("RandomForest:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

###Make the prediction
best_estimator = grid_search.best_estimator_
pred = best_estimator.predict(features)

### Import the required functions from sklearn.metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Compute accuracy
accuracy = accuracy_score(labels, pred)
print("Accuracy: ", accuracy)

### Compute precision
precision = precision_score(labels, pred)
print("Precision: ", precision)

### Compute recall
recall = recall_score(labels, pred)
print("Recall: ", recall)

### Compute F1-score
f1 = f1_score(labels, pred)
print("F1-score: ", f1)

### Now we are trying SVC

### Import required package
from sklearn.svm import SVC

# Define the parameter grid for the SVC
param_grid = {'classifier__C': [0.1, 1, 10],
              'classifier__kernel': ['linear', 'rbf', 'poly'],
              'classifier__degree': [2,3,4],
              'classifier__gamma': ['scale', 'auto']}
              #'selector__k': [2,4,6,8,10]}

print("Create classifier in a pipeline")
#Create the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=8)),
    ('classifier', SVC())
])


print("Create grid search")
### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(pipe, param_grid,cv=cv,scoring='f1')

print("Fit the grid search")
### Fit the grid search object to the data
grid_search.fit(features, labels)

print("SVC:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
#

#Make the prediction using the parameters found by the grid search
best_estimator = grid_search.best_estimator_
pred = best_estimator.predict(features)


### Compute accuracy
accuracy = accuracy_score(labels, pred)
print("Accuracy: ", accuracy)

### Compute precision
precision = precision_score(labels, pred)
print("Precision: ", precision)

### Compute recall
recall = recall_score(labels, pred)
print("Recall: ", recall)

### Compute F1-score
f1 = f1_score(labels, pred)
print("F1-score: ", f1)

### Now we will try AdaBoost

### Import the necessary files
from sklearn.ensemble import AdaBoostClassifier

#Create the pipeline
pipe = Pipeline([
    #('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', AdaBoostClassifier())
])

### Define the parameter grid for the AdaBoostClassifier
param_grid = {'classifier__n_estimators': [10, 50, 100, 200,500],
              'classifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
              'classifier__algorithm': ['SAMME', 'SAMME.R']}
              #'selector__k': [2,4,6,8,10]}

print("Create grid search")
### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1')

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

print("Naive Bayes")

### Import the necessary files
from sklearn.naive_bayes import GaussianNB

###Set the classifier
clf = GaussianNB()

###Fit the classifier
clf.fit(features_train, labels_train)

###Make the prediction
pred = clf.predict(features_test)

### Print the accuracy, precision, recall, and f1-score
print("Accuracy: ", accuracy_score(labels_test, pred))
print("Precision: ", precision_score(labels_test, pred))
print("Recall: ", recall_score(labels_test, pred))
print("F1-score: ", f1_score(labels_test, pred))

###This time, all of our classifiers performed above the mark.
### It looks like we have a winner! We will use the SCV classifier
### Write the classifier here with its optimal parameters:
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('classifier', SVC(C=10,degree=2, gamma='scale', kernel='poly'))
    #('classifier', SVC(C=10, degree=2, gamma='scale',kernel='linear'))
])

#clf = RandomForestClassifier(criterion='entropy', max_depth=5,min_samples_leaf=1,min_samples_split=2,n_estimators=10)
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees', 'bonus_to_salary'] # You will need to use more features

###Here, we use the official test_classifier function to check our results.
test_classifier(pipe, my_dataset, features_list)
























### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipe, my_dataset, features_list)