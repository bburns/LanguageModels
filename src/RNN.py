
# Word Prediction Using RNNs
# Read texts, train an RNN, plot results, and generate sentences.
 

# Set Parameters

TRAIN_AMOUNT = 1.0
NEPOCHS = 10
LAYERS = 2
DROPOUT = 0.1
NVOCAB = 10000
EMBEDDING_DIM = 50
NHIDDEN = EMBEDDING_DIM
N = 5
RNN_CLASS_NAME = 'GRU'
BATCH_SIZE = 32
INITIAL_EPOCH = 0 # to continue training
TRAINABLE = False # train word embedding matrix? if True will slow down training ~2x
PATIENCE = 3 # stop after this many epochs of no improvement
LOSS_FN = 'sparse_categorical_crossentropy'
OPTIMIZER = 'adam'
NVALIDATE = 10000
NTEST = 10000

# these are less likely to be changed
SEED = 0
BASE_DIR = '..'
TEXT_DIR = BASE_DIR + '/data/gutenbergs'
GLOVE_DIR = BASE_DIR + '/_vectors/glove.6B'
GLOVE_FILE = GLOVE_DIR + '/glove.6B.%dd.txt' % EMBEDDING_DIM
MODEL_DIR = BASE_DIR + '/models/gutenbergs'
MODEL_FILE = MODEL_DIR + "/model-train_amount-%s-nvocab-%d-embedding_dim-%d-nhidden-%d-n-%d.h5" % \
             (TRAIN_AMOUNT, NVOCAB, EMBEDDING_DIM, NHIDDEN, N)
print(MODEL_FILE)
import os
os.makedirs(MODEL_DIR, exist_ok=True)


# Import

print('importing libraries ~10sec...')

import sys
print(sys.version)
import os.path
import random
random.seed(SEED)
import re
import heapq

import numpy as np
np.random.seed(SEED)
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except:
    plt = None

import nltk
from nltk import tokenize

from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation, Dropout
from keras.models import Model
from keras.models import Sequential
#from keras.models import load_model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy

print('done')

# define RNN class
rnn_classes = {'SimpleRNN':SimpleRNN, 'LSTM':LSTM, 'GRU':GRU}
RNN_CLASS = rnn_classes[RNN_CLASS_NAME]


# Read Text

# read texts ~ 0.2sec
print('reading texts...')
text = ''
for filename in sorted(os.listdir(TEXT_DIR)):
    filepath = TEXT_DIR +'/' + filename
    if os.path.isfile(filepath) and filename[-4:]=='.txt':
        print(filepath)
        encoding = 'utf-8'
        #with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
        with open(filepath, 'r', encoding=encoding, errors='surrogateescape') as f:
            s = f.read()
            s = s.replace('\r\n','\n')
            s = s.replace('“', '"') # nltk tokenizer doesn't recognize these windows cp1252 characters
            s = s.replace('”', '"')
            text += s
print('done')


# split text into paragraphs, shuffle, and recombine ~0.2sec
paragraphs = re.split(r"\n\n+", text)
print('nparagraphs',len(paragraphs)) # 22989
random.seed(SEED)
random.shuffle(paragraphs)
text = '\n\n'.join(paragraphs)
del paragraphs
print(text[:1000]) # show sample text


# Tokenize Text

print('tokenizing text ~15sec...')
tokens = tokenize.word_tokenize(text.lower())
print('done')

print('first tokens',tokens[:100])


# find the top NVOCAB-1 words ~1sec

token_freqs = nltk.FreqDist(tokens)
token_counts = token_freqs.most_common(NVOCAB-1)

index_to_token = [token_count[0] for token_count in token_counts]
index_to_token.insert(0, '') # oov/unknown at position 0
token_to_index = dict([(token,i) for i,token in enumerate(index_to_token)])

print('start of index_to_token',index_to_token[:10])


# convert words to iwords, ignoring oov (out of vocabulary) words ~1 sec
sequence = []
for token in tokens:
    itoken = token_to_index.get(token)
    if itoken:
        sequence.append(itoken)
nelements = len(sequence)
sequence = np.array(sequence, dtype=np.int)

print('start of token sequence',sequence[:100])

word_to_iword = token_to_index
iword_to_word = {iword:word for iword,word in enumerate(index_to_token)}


# print some info

print('nelements',nelements) # the one million words
print(sequence[:100]) # sample of tokens
print('unique tokens in tokenized text', len(word_to_iword)) # eg 190,000
print('word "the" =', word_to_iword['the'])
iperiod = word_to_iword['.']
print('token ".":',iperiod)
print('iword 99 =',iword_to_word[99])

# print most common and least common words
for i in range(1,10):
    print(i,iword_to_word[i])
nunique = len(word_to_iword)
for i in range(nunique-1, nunique-10, -1):
    print(i,iword_to_word[i])

words = sorted(list(word_to_iword.keys()))
print('first words in dictionary',words[:100])
print('sample words in dictionary',random.sample(words,100))
del words


# Get Embedding Matrix

print('Reading word vectors ~15sec...')
word_vectors = {}
with open(GLOVE_FILE, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_vectors[word] = coefs
print('done')

# print some info
# lots of weird words/names in word vector list, since taken from wikipedia - 
# buttonquail, vaziri, balakirev, 41, foo.com, podicipedidae, morizet, cedel, formula_75
print('Found %s word vectors.' % len(word_vectors))
print('Will use a vocabulary of %d tokens' % NVOCAB)
print('token "a":',word_vectors['a'])
print('some words in word vector list:',list(word_vectors.keys())[:10]) 

# build embedding matrix of the top nvocab words ~30ms
nwords = min(NVOCAB, len(word_to_iword))
E = np.zeros((nwords + 1, EMBEDDING_DIM))
for word, iword in word_to_iword.items():
    if iword > NVOCAB:
        continue
    word_vector = word_vectors.get(word)
    # words not found in embedding index will be all zeros
    if word_vector is not None:
        E[iword] = word_vector

print('number of word vectors in matrix E',len(E))
print('example word vector:',E[1])


#. clear some memory
#del text
#del texts
#del word_vectors


# Split Data

# initialize

ntrain_total = nelements - NVALIDATE - NTEST
ntrain = int(ntrain_total * TRAIN_AMOUNT)

print('total training tokens available:',ntrain_total)
print('training tokens that will be used:',ntrain,'(like a %dk textfile)' % int(ntrain*6/1000))
print('validation tokens:', NVALIDATE)
print('test tokens:', NTEST)

def create_dataset(data, noffset, nelements, ncontext):
    """
    Convert a sequence of values into an x,y dataset.
    data - sequence of integers representing words.
    noffset - starting point
    nelements - how much of the sequence to process
    ncontext - size of subsequences
    e.g. create_dataset([0,1,2,3,4,5,6,7,8,9], 2, 6, 3) =>
         ([[2 3 4],[3 4 5],[4 5 6]], [5 6 7])
    """
    dataX, dataY = [], []
    for i in range(noffset, noffset + nelements - ncontext):
        x = data[i:i+ncontext]
        y = data[i+ncontext]
        dataX.append(x)
        dataY.append(y)
    x_batch = np.array(dataX)
    y_batch = np.array(dataY)
    return x_batch, y_batch

# create train, validate, test sets ~ 5sec

print('create train, validate, test sets ~5sec...')
x_train, y_train = create_dataset(sequence, noffset=0, nelements=ntrain, ncontext=N-1)
x_validate, y_validate = create_dataset(sequence, noffset=-NTEST-NVALIDATE, nelements=NVALIDATE, ncontext=N-1)
x_test, y_test = create_dataset(sequence, noffset=-NTEST, nelements=NTEST, ncontext=N-1)
print('done')


# print info
print('train data size',len(x_train))
print('validation data size',len(x_validate)) # NVALIDATE - (N-1)
print('test data size',len(x_test)) # ditto
print('x_train sample',x_train[:5])
print('y_train sample',y_train[:5])


# Build Model

# define the RNN model

model = Sequential()

# word vectors
embedding_layer = Embedding(input_dim=NVOCAB+1, output_dim=EMBEDDING_DIM, 
                            input_length=N-1, weights=[E])
model.add(embedding_layer)
model.layers[-1].trainable = TRAINABLE

# hidden RNN layer(s)
if LAYERS==1:
    model.add(RNN_CLASS(NHIDDEN))
    model.add(Dropout(DROPOUT))
elif LAYERS==2:
    model.add(RNN_CLASS(NHIDDEN, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(RNN_CLASS(NHIDDEN))
    model.add(Dropout(DROPOUT))
elif LAYERS==3:
    model.add(RNN_CLASS(NHIDDEN, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(RNN_CLASS(NHIDDEN, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(RNN_CLASS(NHIDDEN))
    model.add(Dropout(DROPOUT))
    
# output layer - convert nhidden to nvocab
model.add(Dense(NVOCAB)) 
#model.add(TimeDistributedDense(NVOCAB)) # q. how different from Dense layer?

# convert nvocab to probabilities - expensive
model.add(Activation('softmax')) 

# compile the model ~ 1 sec
metrics = ['accuracy'] # loss is always the first metric returned from the fit method
model.compile(loss=LOSS_FN, optimizer=OPTIMIZER, metrics=metrics)


# Define Functions

def get_best_iword_probs(probs, k):
    """
    Return the best k words and normalized probabilities from the given probabilities.
    e.g. get_best_iword_probs([[0.1,0.2,0.3,0.4]], 2) => [(3,0.57),(2,0.43)]
    """
    iword_probs = [(iword,prob) for iword,prob in enumerate(probs[0])]
    # convert list to a heap, find k largest values
    best_iword_probs = heapq.nlargest(k, iword_probs, key=lambda pair: pair[1])
    # normalize probabilities
    total = sum([prob for iword,prob in best_iword_probs])
    best_normalized_iword_probs = [(iword,prob/total) for iword,prob in best_iword_probs]
    return best_normalized_iword_probs
# test
probs = np.array([[0.1,0.2,0.3,0.4]])
iword_probs = get_best_iword_probs(probs, 2)
print(iword_probs)

def choose_iwords(iword_probs, k):
    """
    Choose k words at random weighted by probabilities.
    eg choose_iwords([(3,0.5),(2,0.3),(9,0.2)], 2) => [3,9] 
    """
    iwords_all = [iword for iword,prob in iword_probs]
    probs = [prob for iword,prob in iword_probs]
    #. choose without replacement?
    iwords = np.random.choice(iwords_all, k, probs) # weighted choice
    return iwords
# test
print(choose_iwords([(3,0.5),(2,0.3),(9,0.2)], 2))


#. make stochastic beam search
#. when have punctuation, start with period 
#. stop when reach a period or max words
#. ->generate_sentence
#. k->beam_width
def generate_text(model, nwords=10, k=5):
    """
    Generate text from the given model with semi stochastic search.
    """
    x = np.zeros((1,N-1), dtype=int)
    # iword = 0
    iword = iperiod
    words = []
    for i in range(nwords):
        x = np.roll(x,-1) # flattens array, rotates to left, and reshapes it
        x[0,-1] = iword # insert new word
        probs = model.predict_proba(x, verbose=0)
        iword_probs = get_best_iword_probs(probs, k)
        iwords = choose_iwords(iword_probs, 1) # choose randomly
        iword = iwords[0]
        word = iword_to_word[iword]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


# Train Model

# define callbacks

class Print_Sentence(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        sentence = generate_text(self.model)
        print('Epoch %d generated text:' % epoch, sentence)

#class BatchRecorder(Callback):
#    def on_train_begin(self, logs={}):
#        self.data = []
#    def on_batch_end(self, batch, logs={}):
#        row = [batch, logs.get('loss'), logs.get('acc')]
#        self.data.append(row)

print_sentence = Print_Sentence()
checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_acc', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_acc', patience=PATIENCE)
#batch_recorder = BatchRecorder()

callbacks = [print_sentence, checkpoint, early_stopping]


print('training model...')
try:
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NEPOCHS, 
                        validation_data=(x_validate, y_validate),
                        callbacks=callbacks)
except KeyboardInterrupt:
    pass

print('Final epoch generated text:', generate_text(model))
print()


#. convert to pandas table
#print(batch_recorder.data)


# Evaluate Model

#model.evaluate(x_test)

#. calculate perplexity - use model.predict_proba()

# is this right? ask on stacko? do calcs for simple case?
print('final perplexity',np.exp(history.history['val_loss']))


# Generate Text

nsentences = 10
nwords_to_generate = 20
k = 10
for i in range(nsentences):
    print(generate_text(model, nwords_to_generate, k))


# Plot Results
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

# Visualize Embeddings
if plt:

    from sklearn.decomposition import PCA

    words = 'alice rabbit mouse said was fell small white gray'.split()
    print('words',words)
    iwords = [word_to_iword[word] for word in words]
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



