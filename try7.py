#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:53:33 2022

@author: oalakkad
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN

def load_dataset():
    data_file = open('dialects_dataframe.p', 'rb')
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
    x_train, x_dev, y_train, y_dev= train_test_split(inputs, labels, stratify=labels, test_size=0.20, random_state=42)

    x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    x_dev, y_dev = SMOTE().fit_resample(x_dev, y_dev)

    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train)

    tf_transformer = TfidfTransformer(use_idf=True).fit(x_train_counts)
    x_train_tf = tf_transformer.transform(x_train_counts).toarray()

    scikit_SVC = LinearSVC()

    clf = scikit_SVC.fit(x_train_tf, y_train)

    x_dev_counts = count_vect.transform(x_dev)
    x_dev_tfidf = tf_transformer.transform(x_dev_counts).toarray()

    predicted = clf.predict(x_train_tf)

    encode = LabelEncoder()
    encode.fit(['syr','leb','pal','jord'])
    y_train_encoded = encode.transform(y_train)
    pred_encoded = encode.transform(predicted)

    train_accuracy = accuracy_score(y_train_encoded, pred_encoded)
    train_precision = precision_score(y_train_encoded, pred_encoded,average='weighted')
    train_recall = recall_score(y_train_encoded, pred_encoded,average='weighted')
    train_f1 = f1_score(y_train_encoded, pred_encoded,average='weighted')
    print("Accuracy on the train set = ", train_accuracy)
    print("Precision of the train set = ", train_precision)
    print("Recall of the train set = ", train_recall)
    print("F1_score of the train set = ", train_f1)

    predicted = clf.predict(x_dev_tfidf)
    encode = LabelEncoder()
    encode.fit(['syr','leb','pal','jord'])
    y_dev_encoded = encode.transform(y_dev)
    pred_encoded = encode.transform(predicted)

    test_accuracy = accuracy_score(y_dev_encoded, pred_encoded)
    test_precision = precision_score(y_dev_encoded, pred_encoded,average='weighted')
    test_recall = recall_score(y_dev_encoded, pred_encoded,average='weighted')
    test_f1 = f1_score(y_dev_encoded, pred_encoded,average='weighted')
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
    d.to_csv(f'SVC_results.csv')
