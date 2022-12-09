#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:28:28 2022

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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.layers import GRU, Input,Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM, Flatten, Conv2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from keras import models, layers
import keras
import tkseem as tk

def load_dataset():
    data_file = open('dialects_dataframe3.p', 'rb')
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

def tokenize_dataset(dataset):
    
    tokenizer = tk.WordTokenizer()
    tokenizer.train('Data/jord.txt')
    X = tokenizer.tokenize(x_train[0])
    tokenizer.encode('Data/jord.txt')
    
    return X

def tokenize_sentences(data, max_words, max_len):
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(data)
    sequences = tok.texts_to_sequences(data)
    sequences_matrix = pad_sequences(sequences,maxlen=max_len)

    return sequences_matrix,tok


if __name__ == "__main__":

    data = load_dataset()
    inputs, labels = create_inputs_labels(data)
    x_train, x_dev, y_train, y_dev= train_test_split(inputs, labels, stratify=labels, test_size=0.20, random_state=42)
    
    num_classes = 4
    encode = LabelEncoder()
    encode.fit(['syr','leb','pal','jord'])
    y_train_encoded = encode.transform(y_train)

    train_characters = tokenize_dataset(x_train)
    train_characters = sorted(list(train_characters))
    num_train_tokens = len(train_characters)
    max_train_length = max([len(txt) for txt in x_train])

    tokenize_train,tok = tokenize_sentences(x_train, num_train_tokens, max_train_length)

    sequences = tok.texts_to_sequences(x_dev)
    tokenize_dev = pad_sequences(sequences,maxlen=max_train_length)
    y_dev_encoded = encode.transform(y_dev)

    tokenize_train = tokenize_train.reshape(tokenize_train.shape[0],tokenize_train.shape[1])
    tokenize_dev = tokenize_dev.reshape(tokenize_dev.shape[0],tokenize_dev.shape[1])
    
    tokenize_train, y_train_encoded = SMOTE().fit_resample(tokenize_train, y_train_encoded)
    tokenize_dev, y_dev_encoded = SMOTE().fit_resample(tokenize_dev, y_dev_encoded)

    scikit_SVC = LinearSVC(max_iter = 10000)

    clf = scikit_SVC.fit(tokenize_train, y_train_encoded)

    predicted = clf.predict(tokenize_train)
    pred_encoded = encode.transform(predicted)

    train_accuracy = accuracy_score(y_train_encoded, pred_encoded)
    train_precision = precision_score(y_train_encoded, pred_encoded,average='weighted')
    train_recall = recall_score(y_train_encoded, pred_encoded,average='weighted')
    train_f1 = f1_score(y_train_encoded, pred_encoded,average='weighted')
    print("Accuracy on the train set = ", train_accuracy)
    print("Precision of the train set = ", train_precision)
    print("Recall of the train set = ", train_recall)
    print("F1_score of the train set = ", train_f1)

    predicted = clf.predict(tokenize_dev)
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
    d.to_csv(f'SVC_results_over.csv')
