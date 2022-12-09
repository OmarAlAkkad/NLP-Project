#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:28:56 2022

@author: oalakkad
"""
import numpy as np
import pandas as pd
import pickle
import arabic_reshaper
import nltk
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

def load_dataset(name):
    datasets = open(f'Data/{name}.txt', 'r', encoding = "UTF-8").readlines()
    dataset = []
    for line in datasets:
        line = arabic_reshaper.reshape(line)
        dataset.append(line.replace("\n",""))

    return np.array(dataset)

def label_dataset(dataset,name):
    labels = []
    for i in range(len(dataset)):
        labels.append(name)
    return np.array(labels)

def tokenize_dataset(dataset):
    sentences = []
    for sentence in dataset:
        tokens = (nltk.word_tokenize(sentence))
        sentences.append(tokens)

    return sentences


def create_dataframe(data, labels):
    d = {
        'Inputs': data ,
        'Labels': labels
        }
    dataframe = pd.DataFrame(data=d)

    return dataframe

def concat_dataframes(frames):
    return pd.concat(frames, ignore_index=True)

def create_inputs_labels(dataset):
    inputs = []
    labels = []
    for i in range(len(dataset)):
        inputs.append(dataset['Inputs'][i])
        labels.append(dataset['Labels'][i])
    return np.array(inputs), np.array(labels)

def list_tostring(data):
    sentences = []
    for sentence in data:
        string1 = ' '.join(sentence)
        sentences.append(string1)
    return sentences


if __name__ == "__main__":
    names = ['syr','jord','pal','leb']
    for name in names:
        locals()[f'{name}'] = load_dataset(name)

    for name in names:
        locals()[f'labels_{name}'] = label_dataset(locals()[f'{name}'],name)
        
    # for name in names:
    #     locals()[f'{name}'] = tokenize_dataset(locals()[f'{name}'])    
        
    dataframes = []
    for name in names:
        locals()[f'{name}_dataframe'] = create_dataframe(locals()[f'{name}'], locals()[f'labels_{name}'])
        pickle.dump(locals()[f'{name}_dataframe'], open(f'{name}_dataframe.p', 'wb'))
        dataframes.append(locals()[f'{name}_dataframe'])

    dataset = concat_dataframes(dataframes)
    
    inputs, labels = create_inputs_labels(dataset)
    encode = LabelEncoder()
    encode.fit(['syr','leb','pal','jord'])
    labels = encode.transform(labels)
    inputs = inputs.reshape(-1,1)
    labels = labels.reshape(-1,1)
    oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    for i in range(len(names)):
        inputs, labels = oversample.fit_resample(inputs, labels)
    inputs = inputs.reshape(-1,)
    labels = labels.reshape(-1,)
    # inputs = list_tostring(inputs)
    labels = encode.inverse_transform(labels)
    dataframe = create_dataframe(inputs, labels)
    pickle.dump(dataframe, open('dialects_dataframe3.p', 'wb'))


