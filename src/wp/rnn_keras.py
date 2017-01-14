
"""
RNN implemented with Keras
"""

import numpy as np

import nltk
from keras.models import Sequential
# from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

from benchmark import benchmark


# #. move to util.py
# class ShowPrediction(keras.callbacks.Callback):
#     """
#     A callback class to show prediction after n epochs
#     """
#     #. pass in a pandas table and col formats, print last row each step using tabulate?
#     #. pass in model, x, y?
#     def __init__(self, interval=10):
#         self.interval = interval
#     def on_train_begin(self, logs={}):
#         pass
#     # def on_batch_end(self, batch, logs={}):
#     #     if batch % 24 == 0:
#     #         s = get_prediction(self.model, x)
#     #         print(batch,s)
#     # def on_epoch_end(self, epoch, logs={}):
#     def on_epoch_begin(self, epoch, logs={}):
#         if epoch % self.interval == 0:
#             s = get_prediction(self.model, x)
#             # print("{: 4d} {}".format(epoch,s))
#             loss, accuracy = self.model.evaluate(x, y, verbose=0)
#             print("{: 6d} {:5.3f} {:5.3f} {}".format(epoch,loss,accuracy,s))


#. move into Data class
def create_dataset(data, nlookback=1):
    """
    convert an array of values into a dataset matrix
    """
    dataX, dataY = [], []
    for i in range(len(data) - nlookback):
        a = data[i:(i + nlookback)]
        dataX.append(a)
        dataY.append(data[i + nlookback])
    return np.array(dataX), np.array(dataY)

def cutoff(p):
    """
    convert probabilities to hard 0's and 1's
    """
    onehot = np.zeros(p.shape)
    for i in range(len(p)):
        row = p[i]
        # print(row)
        mx = row.max()
        row = row/mx
        row = row.astype('int')
        #. if all 1's, choose one at random to be 1, rest 0
        onehot[i] = row
    return onehot




class RnnKeras():
    """
    """

    #. pass rnn type - SimpleRNN, LSTM, GRU
    def __init__(self, data, train_amount=1.0, n=3, nvocab=1000, nhidden=100, nepochs=10, bptt_truncate=4, name_includes=[]):
        """
        Create an RNN model
        data          - source of training and testing data
        train_amount  - percent or number of training characters to use
        nvocab        - max number of vocabulary words to learn
        nhidden       - number of units in the hidden layer
        nepochs       - number of times to run through training data
        bptt_truncate - backpropagate through time truncation
        name_includes - list of properties to include in model name, eg ['nhidden']
        """
        self.data = data
        self.train_amount = train_amount
        self.nvocab = nvocab
        self.nhidden = nhidden
        self.nepochs = nepochs

        # unsure about these...
        self.n = n #... for now - used in test(). yes i think that's what we want - n-1 is amount of context given to model.
        # self.bptt_truncate = bptt_truncate #. -> ntimestepsmax?
        self.bptt_truncate = n #. -> call it ntimestepsmax? keep separate from n?
        self.seqlength = 10 #. -> call it nelements_per_sequence? instead of chopping up by sentences we'll chop up into sequences of this length

        self.name = "RNN-" + '-'.join([key+'-'+str(self.__dict__[key]) for key in name_includes]) # eg 'RNN-nhidden-10'
        self.filename = '%s/rnn-(train_amount-%s-nvocab-%d-nhidden-%d-nepochs-%d).pickle' \
                         % (data.model_folder, str(train_amount), nvocab, nhidden, nepochs)
        self.trained = False
        self.load_time = None
        self.save_time = None
        self.train_time = None
        self.test_time = None
        self.unknown_token = "UNKNOWN" #. ok?
        self.end_token = "END" #.

        # create the keras model
        print("Create model " + self.name)
        self.model = Sequential()
        self.model.add(SimpleRNN(self.nhidden, input_dim=self.nvocab))
        # self.model.add(GRU(self.nhidden, input_dim=self.nvocab)) #
        self.model.add(Dense(self.nvocab)) #. this isn't part of the RNN already?
        self.model.add(Activation('softmax'))
        # categorical_crossentropy is faster than mean_squared_error
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, force_training=False):
        """
        Train the model and save it, or load from file if available.
        force_training - set True to retrain model (ie don't load from file)
        """

        #. put this back in
        # if force_training==False and os.path.isfile(self.filename):
        #     self.load() # see model.py - will set self.load_time
        # else:

        print("{:6} {:5} {:5} {}".format('epoch','loss','acc','s'))

        self.batch_size = 1
        # self.batch_size = 25  #. much slower convergence than 1 - why? what does this do?
        # self.interval = 10
        self.interval = 1
        print("Training model %s on %s percent/chars of training data..." % (self.name, str(self.train_amount)))
        print("Getting training tokens...")
        with benchmark("Prepared training data"):

            #. move all this into Data class

            #. would like this to memoize these if not too much memory, or else pass in tokens and calc them in Experiment class
            tokens = self.data.tokens('train', self.train_amount) # eg ['a','b','.','END']
            print(tokens)
            # get most common words for vocabulary
            word_freqs = nltk.FreqDist(tokens)
            # print(word_freqs)
            wordcounts = word_freqs.most_common(self.nvocab-1)
            # print(wordcounts)
            self.index_to_word = [wordcount[0] for wordcount in wordcounts]
            self.index_to_word.append(self.unknown_token)
            self.index_to_word.sort() #. just using for alphabet dataset
            print(self.index_to_word)
            self.word_to_index = dict([(word,i) for i,word in enumerate(self.index_to_word)])
            # self.nvocab = len(self.index_to_word) #? already set this? cut off with actual vocab length?
            # print(self.word_to_index)
            # replace words not in vocabulary with UNKNOWN
            # tokens = [token if token in self.word_to_index else unknown_token for token in tokens]
            tokens = [token if token in self.word_to_index else self.unknown_token for token in tokens]
            # replace words with numbers
            itokens = [self.word_to_index[token] for token in tokens]
            print(itokens)

            data = to_categorical(itokens, self.nvocab) # one-hot encoding
            # print('data')
            # print(data)

            # self.nlookback = 2 #.. this will be n, right?
            self.nlookback = 3 #.. this will be n, right?
            x, y = create_dataset(data, self.nlookback)
            # print(x)
            # print(y)

        print("Starting gradient descent...")
        with benchmark("Gradient descent finished") as b:
            # self.model.fit(x, y, nb_epoch=self.nepochs, batch_size=self.batch_size, verbose=0, callbacks=[ShowPrediction(self.interval)])
            self.model.fit(x, y, nb_epoch=self.nepochs, batch_size=self.batch_size, verbose=0)

            # show final prediction for set of sequences in x
            s = self.get_prediction(x)
            print('prediction')
            print(s)


        self.train_time = b.time
        self.trained = True
        # self.train_losses = losses

        #. put back
        # save the model
        # self.save()

    def get_prediction(self, x):
        """
        predict the next word in the sequence
        """
        yprobs = self.model.predict(x) # softmax probabilities
        # print(yprobs)
        yonehot = cutoff(yprobs) # onehot encodings
        # print(yonehot)
        yiwords = [row.argmax() for row in yonehot] # iword values
        # print(yiwords)
        # ywords = [vocab[iword] for iword in yiwords]
        ywords = [self.index_to_word[iword] for iword in yiwords]
        # print(ywords)
        s = ' '.join(ywords)
        # print(s)
        return s




if __name__=='__main__':

    np.random.seed(0)
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    import matplotlib.pyplot as plt
    import pandas as pd
    # from tabulate import tabulate

    from data import Data

    # # Check loss calculations
    # # Limit to 1000 examples to save time
    # print("Expected Loss for random predictions: %f" % np.log(nvocab))
    # print("Actual loss: %f" % model.average_loss(X_train[:1000], y_train[:1000]))

    # # Check gradient calculations
    # # use a smaller vocabulary size for speed
    # grad_check_vocab_size = 100
    # np.random.seed(10)
    # model = RnnModel(nvocab=grad_check_vocab_size, nhidden=10, bptt_truncate=1000)
    # model.gradient_check([0,1,2,3], [1,2,3,4])

    # print('see how long one sgd step takes')
    # np.random.seed(0)
    # model = Rnn(nvocab=nvocab, nhidden=nhidden)
    # with benchmark("Time for one sgd step"):
    #     model.sgd_step(X_train[1], y_train[1], 0.005)

    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    data = Data('alphabet')
    # model = RnnKeras(data, nvocab=30, nhidden=3, nepochs=40, train_amount=100)
    model = RnnKeras(data, nvocab=30, nhidden=10, nepochs=40, train_amount=100)
    model.train(force_training=True)

    # model.test(test_amount=100)
    # print('accuracy',model.test_score)
    # print(util.table(model.test_samples))
    # print()

