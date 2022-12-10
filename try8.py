#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 19:42:32 2022

@author: oalakkad
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN

def load_dataset():
    data_file = open('dialects_dataframe_over.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    return data

def create_inputs_labels(dataset):
    inputs = []
    labels = []
    for i in range(len(dataset)):
        inputs.append(dataset['Inputs'][i])
        labels.append(dataset['Labels'][i])
    return np.array(inputs), np.array(labels)


if __name__ == "__main__":

    data = load_dataset()
    inputs, labels = create_inputs_labels(data)
        
    # label encode outputs
    encode = LabelEncoder()
    encode.fit(['syr','leb','pal','jord'])      
    labels = encode.transform(labels)
        
    x_train, x_dev, y_train, y_dev= train_test_split(inputs, labels, stratify=labels, test_size=0.20, random_state=42)

    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train)

    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    x_train_tf = tf_transformer.transform(x_train_counts).toarray()

    scikit_SVC = MultinomialNB()
    
    clf = scikit_SVC.fit(x_train_tf, y_train)

    predicted = clf.predict(x_train_tf)

    train_accuracy = accuracy_score(y_train, predicted)
    train_precision = precision_score(y_train, predicted,average='weighted')
    train_recall = recall_score(y_train, predicted,average='weighted')
    train_f1 = f1_score(y_train, predicted,average='weighted')
    pickle.dump(train_accuracy,open('train_accuracy.p','wb'))
    pickle.dump(train_precision,open('train_precision.p','wb'))
    pickle.dump(train_recall,open('train_recall.p','wb'))
    pickle.dump(train_f1,open('train_f1.p','wb'))
    print("Accuracy on the train set = ", train_accuracy)
    print("Precision of the train set = ", train_precision)
    print("Recall of the train set = ", train_recall)
    print("F1_score of the train set = ", train_f1)
    
    pickle.dump(clf,open('SVC_model.sav','wb'))
    
    
    
    clf = pickle.load(open('SVC_model.sav','rb'))
    x_dev_counts = count_vect.transform(x_dev)
    x_dev_tf = tf_transformer.transform(x_dev_counts).toarray()
    
    predicted = clf.predict(x_dev_tf)
    
    train_accuracy = pickle.load(open('train_accuracy.p','rb'))
    train_precision = pickle.load(open('train_precision.p','rb'))
    train_recall = pickle.load(open('train_recall.p','rb'))
    train_f1 = pickle.load(open('train_f1.p','rb'))

    test_accuracy = accuracy_score(y_dev, predicted)
    test_precision = precision_score(y_dev, predicted,average='weighted')
    test_recall = recall_score(y_dev, predicted,average='weighted')
    test_f1 = f1_score(y_dev, predicted,average='weighted')
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
    d.to_csv('over_NB_results_false.csv')
