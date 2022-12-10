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
    # pickle.dump(dataset, open('dialects_dataframe2.p', 'wb'))


