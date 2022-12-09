# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:54:07 2022

@author: omars
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
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
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

def load_dataset():
    data_file = open('dialects_dataframe2.p', 'rb')
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

def create_bag_of_words(dataset):
    tokens_list = []
    for sentence in dataset:
        for word in sentence:
            if word not in tokens_list:
                tokens_list.append(word)

    return tokens_list

def tokenize_sentences(data, max_words, max_len):
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(data)
    sequences = tok.texts_to_sequences(data)
    sequences_matrix = pad_sequences(sequences,maxlen=max_len)

    return sequences_matrix,tok

def build_model(input_shape, output_sequence_length, source_vocab_size, output_shape):

    inputs = keras.Input(shape=((input_shape[1],input_shape[2])))
    lstm = (LSTM(256, activation="tanh",return_sequences=False))(inputs)
    # attention = SeqSelfAttention(attention_activation='softmax')(lstm)
    # lstm2 = Bidirectional(LSTM(128, activation = 'tanh'))(lstm)
    dense1 = layers.Dense(512, activation="relu")(lstm)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense3 = layers.Dense(128, activation="relu")(dense2)
    dense4 = layers.Dense(64, activation="relu")(dense3)
    outputs = layers.Dense(output_shape, activation = 'softmax')(dense4)
    model = keras.Model(inputs=inputs, outputs=outputs, name="target_insect")
    learning_rate = 0.0009

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = RMSprop(learning_rate),
                  metrics = ['accuracy'])

    return model


if __name__ == "__main__":

    data = load_dataset()
    inputs, labels = create_inputs_labels(data)
    encode = LabelEncoder()
    encode.fit(['syr','leb','pal','jord'])
    labels_encoded = encode.transform(labels)
    x_train, x_dev, y_train, y_dev= train_test_split(inputs, labels_encoded, stratify=labels, test_size=0.20, random_state=42)

    train_words = create_bag_of_words(x_train)
    train_words = sorted(list(train_words))
    num_train_tokens = len(train_words)
    max_train_length = max([len(txt) for txt in x_train])
    
    tokenize_train,tok = tokenize_sentences(x_train, num_train_tokens, max_train_length)

    sequences = tok.texts_to_sequences(x_dev)
    tokenize_dev = pad_sequences(sequences,maxlen=max_train_length)

    num_classes = 4    
    y_train_encoded = to_categorical(y_train, num_classes)    
    y_dev_encoded = to_categorical(y_dev, num_classes)

    tokenize_train = tokenize_train.reshape(tokenize_train.shape[0],tokenize_train.shape[1],-1)
    tokenize_dev = tokenize_dev.reshape(tokenize_dev.shape[0],tokenize_dev.shape[1],-1)

    model = build_model(tokenize_train.shape, max_train_length, num_train_tokens, y_train_encoded.shape[1])
    model.fit(tokenize_train, y_train_encoded, batch_size=5000, epochs=5, validation_data= (tokenize_dev, y_dev_encoded))
