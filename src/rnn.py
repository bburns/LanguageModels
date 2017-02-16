
"""
Word Prediction Using RNNs
Read texts, train an RNN, plot results, and generate sentences.
"""



# --------------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------------

# ~10sec

print('Importing libraries (~10sec)...')

import sys
print(sys.version)
import os
import os.path
import random
import re
import heapq
from importlib import reload

import numpy as np
import pandas as pd

from nltk import tokenize

#from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except:
    plt = None

# local modules
import data as datamodule
import util


# --------------------------------------------------------------------------------
# Set Parameters
# --------------------------------------------------------------------------------

debug            = 0         # 0 or 1
DATASET          = 'gutenbergs' # dataset to use (gutenbergs, alice1)
TRAIN_AMOUNT     = 0.01      # percent of training data to use (for debugging), 0.0 to 1.0
NEPOCHS          = 5         # number of epochs to train model
LAYERS           = 1         # number of RNN layers, 1 to 3
DROPOUT          = 0.1       # amount of dropout to apply after each layer, 0.0 to 1.0
NVOCAB           = 10000     # number of vocabulary words to use
EMBEDDING_DIM    = 100       # dimension of embedding layer - 50, 100, 200, 300
TRAINABLE        = True      # allow embedding matrix to be trained?
NHIDDEN          = 100       # size of hidden layer(s)
N                = 10        # amount to unfold recurrent network
RNN_CLASS        = GRU       # type of RNN to use - SimpleRNN, LSTM, or GRU
BATCH_SIZE       = 32        # size of batch to use for training
INITIAL_EPOCH    = 0         # to continue training
PATIENCE         = 10        # stop after this many epochs of no improvement
VALIDATION_SPLIT = 0.01      # percent of training data to use for validation (0.01 ~10k tokens)
NTEST            = 10000     # number of tokens to use for testing
OPTIMIZER        = 'adam'    # optimizing algorithm to use (sgd, rmsprop, adam, adagrad, adadelta, adamax, nadam)
INITIALIZER      = 'uniform' # random weight initializer (uniform, normal, lecun_uniform, glorot_uniform [default])
SEED             = 0         # random number seed

# LOSS_FN    = 'categorical_crossentropy' # allows calculation of top_k_accuracy, but requires one-hot encoding y values
LOSS_FN    = 'sparse_categorical_crossentropy'
BASE_DIR   = '.'
GLOVE_DIR  = BASE_DIR + '/_vectors/glove.6B'
GLOVE_FILE = GLOVE_DIR + '/glove.6B.%dd.txt' % EMBEDDING_DIM
MODEL_DIR  = BASE_DIR + '/_models/' + DATASET
MODEL_FILE = MODEL_DIR + "/model-train_amount-%s-nvocab-%d-embedding_dim-%d-nhidden-%d-n-%d.h5" % \
                         (TRAIN_AMOUNT, NVOCAB, EMBEDDING_DIM, NHIDDEN, N)

if debug: print(MODEL_FILE)
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------------
# Get Data
# --------------------------------------------------------------------------------

data = datamodule.Data(DATASET)

data.prepare(nvocab=NVOCAB) # ~15sec to tokenize

# split data into train and test sets
x_train, y_train, x_test, y_test = data.split(n=N, ntest=NTEST,
                                              train_amount=TRAIN_AMOUNT, debug=debug)


# --------------------------------------------------------------------------------
# Get Embedding Matrix
# --------------------------------------------------------------------------------

print('Reading word vectors (~15sec)...')
word_vectors = {}
with open(GLOVE_FILE, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_vectors[word] = coefs

# print some info
# lots of weird words/names in word vector list, since taken from wikipedia -
# buttonquail, vaziri, balakirev, 41, foo.com, podicipedidae, morizet, cedel, formula_75
if debug:
    print('found %s word vectors.' % len(word_vectors))
    print('will use a vocabulary of %d tokens' % NVOCAB)
    print('token "a":',word_vectors['a'])
    print('some words in word vector list:',list(word_vectors.keys())[:10])

# build embedding matrix of the top nvocab words ~30ms
def get_embedding_matrix(data, word_vectors, nvocab):
    nwords = nvocab
    embedding_dim = len(word_vectors['a'])
    E = np.zeros((nwords + 1, embedding_dim))
    for word, iword in data.word_to_iword.items():
        if iword > nvocab:
            continue
        word_vector = word_vectors.get(word)
        # words not found in embedding index will be all zeros
        if word_vector is not None:
            E[iword] = word_vector
    return E
E = get_embedding_matrix(data, word_vectors, NVOCAB)

# clear memory
del word_vectors

if debug:
    print('number of word vectors in matrix E',len(E))
    print('example word vector:',E[1])


# --------------------------------------------------------------------------------
# Build Model
# --------------------------------------------------------------------------------

model = Sequential()

# embedding layer
embedding_layer = Embedding(input_dim=NVOCAB+1, output_dim=EMBEDDING_DIM,
                            input_length=N-1, weights=[E])
model.add(embedding_layer)
model.layers[0].trainable = TRAINABLE

# hidden RNN layer(s)
if LAYERS==1:
    model.add(RNN_CLASS(NHIDDEN, init=INITIALIZER))
    model.add(Dropout(DROPOUT))
elif LAYERS==2:
    model.add(RNN_CLASS(NHIDDEN, init=INITIALIZER, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(RNN_CLASS(NHIDDEN, init=INITIALIZER))
    model.add(Dropout(DROPOUT))
elif LAYERS==3:
    model.add(RNN_CLASS(NHIDDEN, init=INITIALIZER, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(RNN_CLASS(NHIDDEN, init=INITIALIZER, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(RNN_CLASS(NHIDDEN, init=INITIALIZER))
    model.add(Dropout(DROPOUT))

# output layer - convert nhidden to nvocab
model.add(Dense(NVOCAB))

# convert nvocab to probabilities - expensive
model.add(Activation('softmax'))

# compile the model ~ 1 sec
metrics = ['accuracy'] # loss is always the first metric returned from the fit method
model.compile(loss=LOSS_FN, optimizer=OPTIMIZER, metrics=metrics)


# --------------------------------------------------------------------------------
# Train Model
# --------------------------------------------------------------------------------

# define callbacks

class Print_Sentence(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        sentence = util.generate_text(self.model, data, N)
        util.uprint('Epoch %d generated text:' % epoch, sentence)

print_sentence = Print_Sentence()
checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_acc', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_acc', patience=PATIENCE)

callbacks = [print_sentence, checkpoint, early_stopping]


print('Training model...')
try:
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NEPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=callbacks)
except KeyboardInterrupt:
    pass

util.uprint('Final epoch generated text:', util.generate_text(model, data, N))
print()

print('history')
print(history.history)


# --------------------------------------------------------------------------------
# Evaluate Model
# --------------------------------------------------------------------------------


# Generate Text

print('generated text:')
nsentences = 10
nwords_to_generate = 20
k = 3
for i in range(nsentences):
    util.uprint(util.generate_text(model, data, N, nwords_to_generate, k))

# calculate test accuracy on the heldout test data
loss, accuracy = model.evaluate(x_test, y_test, BATCH_SIZE, verbose=0)
print("Test loss:",loss)
print("Test accuracy:",accuracy)


# --------------------------------------------------------------------------------
# Plot Results
# --------------------------------------------------------------------------------
if plt:

    # plot loss vs epoch
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('epoch-1')
    plt.ylabel('loss')
    plt.title("Training and Validation Loss vs Epoch")
    plt.legend()
    plt.show()

    # plot accuracy vs epoch
    plt.plot(history.history['acc'], label='Training')
    plt.plot(history.history['val_acc'], label='Validation')
    plt.xlabel('epoch-1')
    plt.ylabel('accuracy')
    plt.title("Training and Validation Accuracy vs Epoch")
    plt.legend()
    plt.show()


# --------------------------------------------------------------------------------
# Visualize Embeddings
# --------------------------------------------------------------------------------
if plt:

    from sklearn.decomposition import PCA

    words = 'alice rabbit mouse said was fell small white gray'.split()
    print('words',words)
    iwords = [data.word_to_iword[word] for word in words]
    print('iwords',iwords)
    vecs = [E[iword] for iword in iwords]
    print('word embedding for alice',vecs[1])

    # now want to reduce dims of these vectors
    pca = PCA(n_components=2)
    pca.fit(vecs)
    vecnew = pca.transform(vecs)
    print('some projections',vecnew[:3])

    # now plot the new vectors with labels
    x = [vec[0] for vec in vecnew]
    y = [vec[1] for vec in vecnew]
    plt.scatter(x, y)

    for i, word in enumerate(words):
        plt.annotate(word, (x[i]+0.1,y[i]+0.1))

    plt.title("Word embeddings projected to 2D")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()



