# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 21:01:39 2022

@author: omars
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tqdm import tqdm
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input,Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM, Flatten, Conv2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D, LSTM, BatchNormalization
from keras.layers import Embedding
from keras.optimizers import Adam,RMSprop
from keras.losses import sparse_categorical_crossentropy
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import OneHotEncoder


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

def tokenize(chars):
    tokened = {}
    for i, char in enumerate(chars):
        tokened[char] = i+1
    tokened[''] = 0
    return tokened

def tokenize_sentences(sentences, indexes):
    tokenized_all = []
    for sentence in sentences:
        tokenized = []
        for word in sentence:
            for char in word:
                tokenized.append(indexes[char])
        tokenized_all.append(tokenized)
    return tokenized_all

def pad(sentences, length):
    return pad_sequences(sentences, maxlen = length, padding = 'post')

def preprocess(text, chars, max_len):
    token_index = tokenize(chars)

    tokenized_text = tokenize_sentences(text, token_index)

    padded_text = pad(tokenized_text, max_len)

    padded_text = padded_text.reshape((padded_text.shape[0],padded_text.shape[1],1))

    return padded_text,token_index

def decode(logits, tokenizer):

    return ' '.join([list(tokenizer.values()).index(prediction) for prediction in np.argmax(logits, 1)])

def build_model(input_shape, output_sequence_length, source_vocab_size, target_vocab_size):

    model = Sequential()
    model.add(Embedding(input_dim=source_vocab_size,output_dim=256,input_length=input_shape[1]))
    model.add(Bidirectional(LSTM(256, activation="tanh",return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='softmax'))
    # model.add((LSTM(512, activation = 'tanh')))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add((Dense(target_vocab_size,activation='softmax')))
    learning_rate = 0.009

    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = RMSprop(learning_rate),
                  metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    data = load_dataset()
    inputs, labels = create_inputs_labels(data)
    x_train, x_dev, y_train, y_dev= train_test_split(inputs, labels, stratify=labels, test_size=0.20, random_state=42)
    characters = set()
    for sentence in inputs:
        for word in sentence:
            for char in word:
                if char not in characters:
                    characters.add(char)

    characters = sorted(list(characters))
    num_source_tokens = len(characters)
    max_sentence_length = max([len(txt) for txt in inputs])

    print("Number of samples:", len(inputs))
    print("Number of unique source tokens:", num_tokens)
    print("Max sequence length for sources:", max_sentence_length)

    processed_inputs, tokens = preprocess(inputs, characters, max_sentence_length)

    model = build_model(x_train.shape, max_sentence_length, num_source_tokens, num_target_tokens)

    model.fit(x_train, y_train, batch_size=8, epochs=1, validation_data=(x_dev,y_dev))
    model.save('test')

    predicted_sentence = model.predict(processed_source[:10])
    decode(predicted_sentence,target_tokens)