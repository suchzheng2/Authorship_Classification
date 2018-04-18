from __future__ import print_function
import os
import numpy as np
import time

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Activation
from keras.layers import Convolution1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from keras.layers import Input, Dropout
from keras.optimizers import SGD, Adadelta
from keras.models import Sequential
from sklearn.model_selection import train_test_split, KFold
import csv
import sys
import codecs
import matplotlib.pyplot as plt
import datetime as dt
from csv import reader
csv.field_size_limit(sys.maxsize)
global MAX_SEQUENCE_LENGTH, MAX_NB_WORDS , EMBEDDING_DIM, VALIDATION_SPLIT, DROP_OUT, Nb_EPOCH, BATCH_SIZE, Classes 
global Classes, DROP_OUT, EMBEDDING_DIM, Nb_EPOCH, FILENAME, TEXT_DATA_DIR

MAX_SEQUENCE_LENGTH = 1000000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
DROP_OUT = 0.3
Nb_EPOCH = 3
BATCH_SIZE = 10
Classes = 2

parameters = {
"classes" : [2],
#"batches" : [10, 20, 50, 100],
#"epochs": [1, 10, 25, 50, 100], 
#"dropout_rate" : [0.0, 0.1, 0.2, 0.3, 0.4],
#"embedding_dimension" : [25, 50, 100, 200]
}


GLOVE_DIR = '/Users/suzy/Documents/git/twitter-sentiment-analysis/glove/'
FILENAME = 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt'
TEXT_DATA_DIR = '/Users/suzy/Documents/GuidedStudy/author/dataset/Author_14.csv'
global embeddings
embeddings = {}
fname = os.path.join(GLOVE_DIR, FILENAME)
f = codecs.open(fname, 'r', encoding='utf-8')

for line in f:
    values = line.split()
    if(len(values) == 0):
        print('Empty')
    else:
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs

f.close()
print('Found %s word vectors.' % len(embeddings))


print('Processing text dataset')
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

f = codecs.open(TEXT_DATA_DIR, 'r', encoding='utf-8')

r = reader(TEXT_DATA_DIR)
i = 0
with open(TEXT_DATA_DIR,'rb') as csvfile:
    spamreader = reader(csvfile,delimiter=',')
    raw_data_list= list(spamreader)
    sorted(raw_data_list)

for row in raw_data_list:
    label_id = len(labels_index)
    labels_index[int(row[0])] = label_id
    labels.append(label_id)
    texts.append(row[1])
print('Found %s texts.' % len(texts))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


print('Preparing embedding matrix.')

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Training model.')
model = Sequential()

model.add(Embedding(                          # Layer 0, Start
    input_dim=nb_words + 1,                   # Size to dictionary, has to be input + 1
    output_dim=EMBEDDING_DIM,                 # Dimensions to generate
    weights=[embedding_matrix],               # Initialize word weights
    input_length=MAX_SEQUENCE_LENGTH))        # Define length to input sequences in the first layer

model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(18))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Done compiling.")

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=1, batch_size=150)

