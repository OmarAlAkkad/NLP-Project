#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:15:12 2022

@author: oalakkad
"""
import numpy as np
import pandas as pd
import pickle
import arabic_reshaper
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import nltk

def load_dataset(name):
    '''
    takes name of dataset as input.
    Loads dataset into a list.
    returns list.
    '''
    datasets = open(f'Data/{name}.txt', 'r', encoding = "UTF-8").readlines()
    dataset = []
    for line in datasets:
        line = arabic_reshaper.reshape(line) #reshapes string from right to left into memory
        dataset.append(line.replace("\n",""))

    return np.array(dataset)

def label_dataset(dataset,name):
    '''
    Parameters
    ----------
    dataset : array of strings
        array containing dataset used for creating models.
    name : string
        label of each string.

    Returns
    -------
    numpy array
        array containing labels generated.
    '''
    labels = []
    for i in range(len(dataset)):
        labels.append(name)
    return np.array(labels)

def create_dataframe(data, labels):
    '''
    

    Parameters
    ----------
    data : array of strings
        array containing input data.
    labels : array of strings
        array containing labels of dataset.

    Returns
    -------
    dataframe : pandas dataframe
        pandas dataframe of inputs and labels to ease processing.

    '''
    d = {
        'Inputs': data ,
        'Labels': labels
        }
    dataframe = pd.DataFrame(data=d)

    return dataframe

def concat_dataframes(frames):
    '''
    

    Parameters
    ----------
    frames : List of lists
        lists of all the dataframes generated.

    Returns
    -------
    pandas dataframe
        concatenated dataframe of all the lists in the input.

    '''
    return pd.concat(frames, ignore_index=True)

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

def create_oversampled_data(dataset):
    '''
    
    Parameters
    ----------
    dataset : pandas dataframe
        dataframe containing inputs and labels.

    Returns
    -------
    dataframe : pandas dataframe
        dataframe containing oversampled datapoints using the random over sampler function.

    '''
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
    
    return dataframe

def create_undersampled_data(dataset):
    '''

    Parameters
    ----------
    dataset : pandas dataframe
        dataframe containing inputs and labels.

    Returns
    -------
    dataframe : pandas dataframe
        dataframe containing undersampled datapoints using the random under sampler function.

    '''
    inputs, labels = create_inputs_labels(dataset)
    encode = LabelEncoder()
    encode.fit(['syr','leb','pal','jord'])
    labels = encode.transform(labels)
    inputs = inputs.reshape(-1,1)
    labels = labels.reshape(-1,1)
    undersample = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    for i in range(len(names)):
        inputs, labels = undersample.fit_resample(inputs, labels)
    inputs = inputs.reshape(-1,)
    labels = labels.reshape(-1,)
    labels = encode.inverse_transform(labels)
    dataframe = create_dataframe(inputs, labels)
    
    return dataframe


if __name__ == "__main__":
    names = ['syr','jord','pal','leb']
    for name in names:
        locals()[f'{name}'] = load_dataset(name)

    for name in names:
        locals()[f'labels_{name}'] = label_dataset(locals()[f'{name}'],name)

    dataframes = []
    for name in names:
        locals()[f'{name}_dataframe'] = create_dataframe(locals()[f'{name}'], locals()[f'labels_{name}'])
        pickle.dump(locals()[f'{name}_dataframe'], open(f'{name}_dataframe.p', 'wb'))
        dataframes.append(locals()[f'{name}_dataframe'])

    dataset = concat_dataframes(dataframes)
    pickle.dump(dataset, open('dialects_dataframe_normal.p', 'wb'))
    
    dataframe = create_oversampled_data(dataset)
    pickle.dump(dataframe, open('dialects_dataframe_over.p', 'wb'))
    
    dataframe = create_undersampled_data(dataset)
    pickle.dump(dataframe, open('dialects_dataframe_under.p', 'wb'))
