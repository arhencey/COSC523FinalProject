"""
Created on Fri Dec  3 21:00:27 2021

@author: Talha Ahmed
"""
import numpy as np
import pandas as pd 
import os
import gc
import sys
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, StratifiedKFold
# import xgboost as xgb

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

# from transformers import AutoModel, AutoTokenizer
import json
from tensorflow.keras.models import load_model
import re
import pandas as pd
import string
import keras
from sklearn.svm import SVR
import math
import pickle
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split 
# from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM,Dropout,concatenate
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Activation, Embedding, LSTM,Dropout,Bidirectional,GRU
# from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Flatten ,Embedding,Input,Conv1D,GlobalAveragePooling1D,GlobalMaxPooling1D,Dropout,MaxPooling1D,Bidirectional,GRU,Concatenate
from keras.models import Sequential,Model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def svmelmo():
    maindata = pd.read_csv('train.csv')
    text_target=maindata['target']
    text_excerpt=maindata['excerpt']
    
    pickle_in = open("elmo_train_03122021.pickle", "rb")
    main_embedding=pickle.load(pickle_in)
    
    #Splitting the data into training and testing split
    X_train,X_test,y_train,y_test,train_embedding,test_embedding= train_test_split(text_excerpt,
                                                      text_target,main_embedding,
                                                      test_size=0.30)
    
    target = y_train.to_numpy()
    
    def rmse_score(y_true,y_pred):
        return np.sqrt(mean_squared_error(y_true,y_pred))
    
    config = {
        'batch_size': 128,
        'max_len': 256,
        'seed': 42,
    }
    
    mean_scores=[]
    def get_preds(X,y,X_test,nfolds=20,C=10,kernel='rbf'):
        scores = list()
        preds = np.zeros((X_test.shape[0]))
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=config['seed'])
        for k, (train_idx, valid_idx) in enumerate(kf.split(X_train)): 
            model = SVR(C=C,kernel=kernel,gamma='auto')
            train_x,train_y,val_x,val_y=X[train_idx], y[train_idx],X[valid_idx], y[valid_idx]
            
            
            model.fit(train_x,train_y)
            prediction = model.predict(val_x)
            score = rmse_score(prediction,val_y)
            print(f'Fold {k} , rmse score: {score}')
            scores.append(score)
            preds += model.predict(X_test)
            
        print("mean rmse",np.mean(scores))
        mean_scores.append(np.mean(scores))
        return np.array(preds)/nfolds
    
    preds1 = get_preds(train_embedding,target,test_embedding)