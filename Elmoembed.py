# Only works with tf 1x (Only run this code if you have this otherwise it wont run on tf 2.0 or higher)
"""
Created on Fri Dec  3 16:17:45 2021

@author: Talha Ahmed
"""

import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 200)

#Reading in the csv file
maindata=pd.read_csv('train.csv')
text_target=maindata['target']
text_excerpt=maindata['excerpt']

#Splitting the data into training and testing split
X_train,X_test,y_train,y_test = train_test_split(text_excerpt,
                                                  text_target,
                                                  test_size=0.30)

# remove URL's from train and test
X_train = X_train.apply(lambda x: re.sub(r'http\S+', '', x))

X_test = X_test.apply(lambda x: re.sub(r'http\S+', '', x))

# remove punctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

X_train = X_train.apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
X_test = X_test.apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
X_train = X_train.str.lower()
X_test = X_test.str.lower()

# remove numbers
X_train = X_train.str.replace("[0-9]", " ")
X_test = X_test.str.replace("[0-9]", " ")

# remove whitespaces
X_train = X_train.apply(lambda x:' '.join(x.split()))
X_test = X_test.apply(lambda x: ' '.join(x.split()))

# import spaCy's language model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

X_train = lemmatization(X_train)
X_test = lemmatization(X_test)

import tensorflow_hub as hub
import tensorflow as tf

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def elmo_vectors(x):
  embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

list_train = [X_train[i:i+100] for i in range(0,X_train.shape[0],100)]
list_test = [X_test[i:i+100] for i in range(0,X_test.shape[0],100)]

# Extract ELMo embeddings
elmo_train = [elmo_vectors(x) for x in list_train]
elmo_test = [elmo_vectors(x) for x in list_test]

elmo_train_new = np.concatenate(elmo_train, axis = 0)
elmo_test_new = np.concatenate(elmo_test, axis = 0)

# save elmo_train_new
pickle_out = open("elmo_train_03122021.pickle","wb")
pickle.dump(elmo_train_new, pickle_out)
pickle_out.close()

# save elmo_test_new
pickle_out = open("elmo_test_03122021.pickle","wb")
pickle.dump(elmo_test_new, pickle_out)
pickle_out.close()