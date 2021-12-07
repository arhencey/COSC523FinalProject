"""
Created on Wed Dec  1 22:33:54 2021

@author: Talha Ahmed
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def basetfidf():
    #Reading in the csv file
    maindata=pd.read_csv('train.csv')
    text_target=maindata['target']
    text_excerpt=maindata['excerpt']
    
    #Splitting the data into training and testing split
    X_train,X_test,y_train,y_test = train_test_split(text_excerpt,
                                                      text_target,
                                                      test_size=0.30)
    #Converting the series into list
    xtrain=list(X_train.to_numpy())
    xtest=list(X_test.to_numpy())
    ytrain=list(y_train.to_numpy())
    ytest=list(y_test.to_numpy())
    
    # Instantiate the Tfidfvectorizer
    tfidf_vectorizer=TfidfVectorizer() 
    
    # Send our docs into the Vectorizer
    tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(xtrain)
    
    # Transpose the result into a more traditional TF-IDF matrix, and convert it to an array.
    X = tfidf_vectorizer_vectors.T.toarray()
    
    # Convert the matrix into a dataframe using feature names as the dataframe index.
    pdframe = pd.DataFrame(X, index=tfidf_vectorizer.get_feature_names()) 
    
    predcomp=np.zeros(len(xtest))
    
    for m in range(len(xtest)):
    
        # Vectorize the test excerpt.
        q = [xtest[m]]
        q_vec = tfidf_vectorizer.transform(q).toarray().reshape(pdframe.shape[0],)
        
        # Calculate cosine similarity.
        sim = {}
        for i in range(len(pdframe.columns)-1):
            sim[i] = np.dot(pdframe.loc[:, i].values, q_vec) / np.linalg.norm(pdframe.loc[:, i]) * np.linalg.norm(q_vec)
        
        # Sort the values 
        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
        
        printlinks=0
        # Print the excerpts and the top similarity value
        for k, v in sim_sorted:
            if v != 0.0:
                if printlinks<1:
                    # print("Excerpt: "+str(m))
                    matchdocnum=k
                    printlinks=1
         
        #Getting the target complexity value predictions
        predcomp[m]=ytrain[matchdocnum]
        
    #Calculating the RMSE between the test target and prediction values
    RMSEval=np.sqrt(mean_squared_error(ytest,predcomp))
    #print("RMSE value for test set: ",RMSEval)
    return RMSEval
