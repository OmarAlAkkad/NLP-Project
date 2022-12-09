#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:02:26 2022

@author: oalakkad
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC

def load_dataset(data_type):
    '''

    Parameters
    ----------
    data_type : String
        specifiy what dataset to load
        datasets can be normal sized, undersampled, and oversampled.

    Returns
    -------
    data : pandas dataframe
        dataset containing inputs and labels.

    '''
    data_file = open(f'dialects_dataframe_{data_type}.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()
    
    return data
    
def create_inputs_labels(dataset):
    '''
    

    Parameters
    ----------
    dataset : pandas dataframe
        dataframe containing list of inputs and labels.

    Returns
    -------
    2 numpy arrays
        numpy arrays containing inputs and labels.

    '''
    inputs = []
    labels = []
    for i in range(len(dataset)):
        inputs.append(dataset['Inputs'][i])
        labels.append(dataset['Labels'][i])
    return np.array(inputs), np.array(labels)

def choose_fitter(method):
    '''
    

    Parameters
    ----------
    method : String
        select type of method to use when generating a model.

    Returns
    -------
    object of selected model
        create an instance of sklearn model depending on input.

    '''
    if method == 'NB':
        return MultinomialNB()
    elif method == 'Logistic':
        return LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
    elif method == 'SVC':
        return LinearSVC(max_iter = 10000)
    
if __name__ == "__main__":
    
    data_types = ['normal','under','over']
    methods = ['NB', 'Logistic','SVC']
    idfs = [True,False]
    
    # loop over types of datasets available
    for data_type in data_types:
        data = load_dataset(data_type)
        inputs, labels = create_inputs_labels(data)
        
        # vectorize inputs
        count_vect = CountVectorizer()
        inputs = count_vect.fit_transform(inputs)
        
        # label encode outputs
        encode = LabelEncoder()
        encode.fit(['syr','leb','pal','jord'])      
        labels = encode.transform(labels)
        
        #train test splits
        x_train, x_dev, y_train, y_dev= train_test_split(inputs, labels, stratify=labels, test_size=0.20, random_state=42)
        
        # loop through choices of idf
        for idf in idfs:
            tf_transformer = TfidfTransformer(use_idf=idf).fit(x_train)
            x_train_tf = tf_transformer.transform(x_train).toarray()
            x_dev_tfidf = tf_transformer.transform(x_dev).toarray()
            
            # loop through methods chosen
            for method in methods:
                regressor = choose_fitter(method)
                clf = regressor.fit(x_train_tf, y_train)
                train_predicted = clf.predict(x_train_tf)
            
                train_accuracy = accuracy_score(y_train, train_predicted)
                train_precision = precision_score(y_train, train_predicted,average='weighted')
                train_recall = recall_score(y_train, train_predicted,average='weighted')
                train_f1 = f1_score(y_train, train_predicted,average='weighted')
                print("Accuracy on the train set = ", train_accuracy)
                print("Precision of the train set = ", train_precision)
                print("Recall of the train set = ", train_recall)
                print("F1_score of the train set = ", train_f1)
            
                test_predicted = clf.predict(x_dev_tfidf)
            
                test_accuracy = accuracy_score(y_dev, test_predicted)
                test_precision = precision_score(y_dev, test_predicted,average='weighted')
                test_recall = recall_score(y_dev, test_predicted,average='weighted')
                test_f1 = f1_score(y_dev, test_predicted,average='weighted')
                print("Accuracy on the dev set = ", test_accuracy)
                print("Precision of the dev set = ", test_precision)
                print("Recall of the dev set = ", test_recall)
                print("F1_score of the dev set = ", test_f1)
            
                d = pd.DataFrame({'Data' : ['train','test'],
                         'Accuracy': [train_accuracy,test_accuracy],
                         'Precision': [train_precision,test_precision],
                         'Recall': [train_recall,test_recall],
                         'F1 Score': [train_f1,test_f1],
                         })
                if idf:
                    d.to_csv(f'{data_type}_{method}_results_true.csv')
                    print(f'Saving ... {data_type}_{method}_results_true')
                else:
                    d.to_csv(f'{data_type}_{method}_results_false.csv')
                    print(f'Saving ... {data_type}_{method}_results_false')
