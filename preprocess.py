# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:34:15 2022

@author: omars
"""
import numpy as np
import pandas as pd
import pickle
import arabic_reshaper

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

    dataframes = []
    for name in names:
        locals()[f'{name}_dataframe'] = create_dataframe(locals()[f'{name}'], locals()[f'labels_{name}'])
        pickle.dump(locals()[f'{name}_dataframe'], open(f'{name}_dataframe.p', 'wb'))
        dataframes.append(locals()[f'{name}_dataframe'])

    dataset = concat_dataframes(dataframes)
    pickle.dump(dataset, open('dialects_dataframe.p', 'wb'))
