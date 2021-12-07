"""
Created on Fri Dec  3 23:23:25 2021

@author: Talha Ahmed
"""
from prepare_data import get_feature_data
import numpy as np
import pandas as pd
import pylab as pl
import sklearn
from sklearn import svm, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# By default, Sklearn forces warnings into your terminal.
# Here, we're writing a dummy function that overwrites the function
# that prints out numerical warnings.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def svmfeats():
    # Read in the mushrooms dataset.
    X,y =get_feature_data()
    
        
    # Use Sklearn to get splits in our data for training and testing.
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
     
    Finetune=1 #Set whether we want Coarse Grid Search (set at 0) or Fine Grid Seach (set at 1) 
    
    if Finetune==0:
        # # # # # # # # # # # # # # # # # # # # #
        # Coarse Grid Search                    #
        #   - Broad sweep of hyperparemeters.   #
        # # # # # # # # # # # # # # # # # # # # #
        
        # Set the parameters by cross-validation
        tuned_parameters = [
            {
                'kernel': ['rbf'], 
                'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
                'C': [1, 10, 100, 1000]
            }
        ]
        
        
        scores = ['neg_root_mean_squared_error']
        
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
        
            clf = GridSearchCV(
                SVR(), tuned_parameters, scoring=score,cv=5
            )
            clf.fit(x_train, y_train)
        
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
        
    else:
        # # # # # # # # # # # # # # # # # # # # #
        # Fine Grid Search                    #
        #   -Fine sweep of hyperparemeters.   #
        # # # # # # # # # # # # # # # # # # # # #
        
        # Set the parameters by cross-validation
        tuned_parameters = [
            {
                'kernel': ['rbf'], 
                'gamma': [1e-1,1e-2,1e-3],
                'C': [1000, 1100, 1200, 1300, 1400]
            }
        ]
        
        
        scores = ['neg_root_mean_squared_error']
        
        for score in scores:
            print("# Tuning hyper-parameters through fine grid search for %s" % score)
            print()
        
            clf = GridSearchCV(
                SVR(), tuned_parameters, scoring=score,cv=5
            )
            clf.fit(x_train, y_train)
        
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
            
            #Computing rmse value
            predcomp=clf.predict(x_test)
            RMSEval=np.sqrt(mean_squared_error(y_test,predcomp))
            print("RMSE value for test set: ",RMSEval)
            return RMSEval
