# This file implements the neural network models to be trained on the dataset

import numpy as np
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, SimpleRNN, Dropout
from keras.layers.embeddings import Embedding
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split

from prepare_data import get_feature_data, get_excerpt_data
from visualize import plot_loss

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def get_features_only_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape, dtype=tf.float32))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(Adam(learning_rate=0.001), loss='mse', metrics=[root_mean_squared_error])
    return model

def get_excerpt_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_length))
    model.add(Flatten())
    #model.add(SimpleRNN(256))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(Adam(learning_rate=0.001), loss='mse', metrics=[root_mean_squared_error])
    return model

def get_rnn_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=max_length, name='embedding_layer'))
    model.add(SimpleRNN(256))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(Adam(learning_rate=0.001), loss='mse', metrics=[root_mean_squared_error])
    return model

def train_features_only():
    print('\n--- Reading data...')
    X, y = get_feature_data()
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    # split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print('\n--- Creating model...')
    model = get_features_only_model(X_train.shape)

    X_train = np.asarray(X_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')

    history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            shuffle=True,
            epochs=25,
    )
    return history

def train_excerpts(vocab_size, max_length):
    print('\n--- Reading data...')
    X, y = get_excerpt_data()
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    # integer encode the documents
    X = np.squeeze(X)
    X = [one_hot(d, vocab_size) for d in X]

    # pad documents to a max length
    X = pad_sequences(X, maxlen=max_length, padding='post')

    # split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print('\n--- Creating model...')
    model = get_excerpt_model(vocab_size, max_length)

    history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            shuffle=True,
            epochs=25,
    )
    return history

def train_rnn():
    print('\n--- Reading data...')
    X, y = get_excerpt_data()
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    # integer encode the documents
    X = np.squeeze(X)
    X = [one_hot(d, vocab_size) for d in X]

    # pad documents to a max length
    X = pad_sequences(X, maxlen=max_length, padding='post')

    # split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print('\n--- Creating model...')
    model = get_rnn_model(vocab_size, max_length)

    history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            shuffle=True,
            epochs=25,
    )
    return history

vocab_size = 10000
max_length = 100
#plot_loss(train_features_only())
#plot_loss(train_excerpts(vocab_size, max_length))
plot_loss(train_rnn())

