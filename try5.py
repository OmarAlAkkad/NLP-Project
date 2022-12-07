# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:59:06 2022

@author: omars
"""
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('arabic'))