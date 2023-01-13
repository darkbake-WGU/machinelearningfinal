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

### We will use every single numeric feature available
features_list = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] # You will need to use more features

### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "rb") as data_file:

### We are also going to convert the data set into a pandas dataframe for use
### in the outlierCleaner() function that I rewrote.
    
data_dict = pickle.load(open("final_project_dataset.pkl", "rb") ) 

###creating dataFrame from dictionary - pandas 
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=float) 

### Describe the data to check that everything is going okay
print( df.describe())
    

### Task 2: Remove outliers
### This outlierCleaner() function has been rewritten

cleaned_data = outlierCleaner(df)


### Task 3: Create new feature(s)

### Create new feature: bonus_to_salary ratio
cleaned_data['bonus_to_salary'] = cleaned_data['bonus'] / cleaned_data['salary']

### Verify that the new feature was added correctly
print(cleaned_data.describe().loc[:,['bonus_to_salary']])

###Check to make sure that the dataframe still looks healthy
print(cleaned_data.describe())

### Only use the columns in features_list
cleaned_data = cleaned_data.filter(items=features_list)

### Task 3B: Apply scaling to the features
### I added this task in as an extra step

###Import files necessary for scaling
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

### Create a StandardScaler object
scaler = StandardScaler()

### Use the imputer on any missing values
### Fit the scaler to the features using fit_transform
### The fit_transform method applies the scaler to the features and returns the scaled features

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
scaled_features = scaler.fit_transform(imp.fit_transform(cleaned_data))

### Convert it to a pandas data frame
scaled_features_df = pd.DataFrame(scaled_features,columns=cleaned_data[features_list[0:]].columns)

###Drop NaN values
scaled_features_df = scaled_features_df.dropna()

### This will check on the health of the data frame once again
print("Scaled_features: ", scaled_features_df.describe())

### Now we must convert the data frame to a dictionary so it can be used
### in featureFormat()
data_dict = scaled_features_df.to_dict(orient='index', into=dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

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

### Here we are splitting the data into testing and training segments
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    

### For some reason, the labels have to be converted to a categorical variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_train = le.fit_transform(labels_train)
labels_test = le.fit_transform(labels_test)

### Import the RandomForestClassifier from sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier

### We are going to be using a grid search method in order to determine
### the best parameters for the RandomForest classifier.

### Import GridSearchCV
from sklearn.model_selection import GridSearchCV

### Define the parameter grid for the grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

### Create a random forest classifier object
clf = RandomForestClassifier(random_state=42)

### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')

### Fit the grid search object to the data
grid_search.fit(features_train, labels_train)

print("RandomForest:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
    
### Create a RandomForestClassifier object using the best parameters
clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_leaf=1, min_samples_split=2,random_state=42)

### Fit the classifier to the data
clf.fit(features_train, labels_train)

### Run a prediction test
pred = clf.predict(features_test)

### Import the required functions from sklearn.metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Compute accuracy
accuracy = accuracy_score(labels_test, pred)
print("Accuracy:", accuracy)

### Compute precision
precision = precision_score(labels_test, pred)
print("Precision:", precision)

### Compute recall
recall = recall_score(labels_test, pred)
print("Recall:", recall)

### Compute F1-score
f1 = f1_score(labels_test, pred)
print("F1-score:", f1)

### Now we are trying SVC

### Import required package
from sklearn.svm import SVC

# Define the parameter grid for the SVC
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf']}

### Create a SVC classifier object
clf = SVC()

### Create a grid search object using the classifier and parameter grid
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')

### Fit the grid search object to the data
grid_search.fit(features_train, labels_train)

print("SVC:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

### Create a SVM classifier object using the best parameters found
clf = SVC(kernel='rbf', C=100)

### Fit the classifier to the training data
clf.fit(features_train, labels_train)

### Run a prediction test
pred = clf.predict(features_test)


### Compute accuracy
accuracy = accuracy_score(labels_test, pred)
print("Accuracy:", accuracy)

### Compute precision
precision = precision_score(labels_test, pred)
print("Precision:", precision)

### Compute recall
recall = recall_score(labels_test, pred)
print("Recall:", recall)

### Compute F1-score
f1 = f1_score(labels_test, pred)
print("F1-score:", f1)

### Now we will try AdaBoost

### Import the necessary files
from sklearn.ensemble import AdaBoostClassifier

### Create an AdaBoostClassifier object
clf = AdaBoostClassifier(random_state=42)

### Define the parameter grid for the AdaBoostClassifier
param_grid = {'n_estimators':[50, 100, 200],
              'learning_rate':[0.1, 0.5, 1.0]}

### Create a GridSearchCV object
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')

### Fit the GridSearchCV object to the data
grid_search.fit(features_train, labels_train)

print("AdaBoost:")

### Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

### Make predictions using the best estimator
pred = grid_search.best_estimator_.predict(features_test)

### Print the accuracy, precision, recall, and f1-score
print("Accuracy:", accuracy_score(labels_test, pred))
print("Precision:", precision_score(labels_test, pred))
print("Recall:", recall_score(labels_test, pred))
print("F1-score:", f1_score(labels_test, pred))

### It looks like we have a winner! We will use the AdaBoostClassifier
### Write the classifier here with its optimal parameters:
clf = AdaBoostClassifier(learning_rate=.5, n_estimators = 200, random_state=42)































### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)