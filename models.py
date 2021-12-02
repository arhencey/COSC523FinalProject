# This file implements the neural network models to be trained on the dataset

import numpy as np
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.layers.embeddings import Embedding
import tensorflow as tf

from prepare_data import get_data

#X_train = training_data[["excerpt"]].to_numpy()
#y_train = training_data[["target"]].to_numpy()
#X_test = testing_data[["excerpt"]].to_numpy()
#y_test = testing_data[["target"]].to_numpy()

def get_features_only_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape, dtype=tf.float32))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    return model

print('\n--- Reading data...')
X_train, y_train = get_data()
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

print('\n--- Creating model...')
model = get_features_only_model(X_train.shape)
model.summary

X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')

history = model.fit(
        X_train,
        y_train,
        shuffle=True,
        epochs=100,
)
