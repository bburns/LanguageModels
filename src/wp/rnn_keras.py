
"""
RNN implemented with Keras
"""

from benchmark import benchmark
with benchmark('import'): # 19 secs cold, 4 secs warm
    import os
    import heapq

    import numpy as np
    import nltk
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    import keras
    from keras.models import Sequential
    from keras.models import load_model
    # from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import SimpleRNN, LSTM, GRU
    from keras.layers import Dense, Activation, Dropout
    from keras.callbacks import EarlyStopping
    from keras.utils.np_utils import to_categorical

    from model import Model
    import util
    import vocab


#. move to util?
#. refactor
def get_relevance(yactual, yprobs, k=3):
    """
    Get relevance score for the given probabilities and actual values.
    ie check if the k most probable values include the actual value.
    eg _____________
    """
    # print(yactual[:3]) # onehot
    yiwords = [row.argmax() for row in yactual] # iword values
    # print(yiwords[:3])
    nrelevant = 0
    ntotal = 0
    for yp, yiw in zip(yprobs, yiwords):
        pairs = [(iword,p) for iword,p in enumerate(yp)]
        # print(pairs[:3])
        best_pairs = heapq.nlargest(k, pairs, key=lambda pair: pair[1])
        # print(best_pairs[:3])
        best_iwords = [pair[0] for pair in best_pairs]
        # print(best_iwords[:3])
        # print(yiw)
        relevant = (yiw in best_iwords)
        nrelevant += int(relevant)
        ntotal += 1
        # # print(self.nvocab)
        # best_words = [(self.m.index_to_word[iword],p) for iword,p in best_iwords]
    relevance = nrelevant/ntotal
    return relevance


#. move to util.py
class ShowLoss(keras.callbacks.Callback):
    """
    A callback class to show loss, accuracy, and prediction during training.
    """
    #. pass in a pandas table and col formats, print last row each step using tabulate?
    def __init__(self, m, x_validate, y_validate, epoch_interval=10):
        self.m = m
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.epoch_interval = epoch_interval
        self.rows = []

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.epoch_interval == 0:
            loss, accuracy = self.model.evaluate(self.x_validate, self.y_validate, verbose=0)

            # s = self.m.get_prediction(self.x_validate)
            # yprobs = self.model.predict(x) # softmax probabilities
            yprobs = self.model.predict(self.x_validate) # softmax probabilities
            # yprobs = rnn.predict(x) # softmax probabilities
            # print(yprobs[:3])
            yonehot = cutoff(yprobs) # onehot encodings
            # print(yonehot)
            yiwords = [row.argmax() for row in yonehot] # iword values
            # print(yiwords[:3])
            # ywords = [vocab[iword] for iword in yiwords]
            # ywords = [self.m.index_to_word[iword] for iword in yiwords]
            ywords = self.m.vocab.get_tokens(yiwords)
            # ywords = [vocab[iword] for iword in yiwords]
            # print(ywords)
            s = ' '.join(ywords)
            # print(s)
            # return s

            relevance = get_relevance(self.y_validate, yprobs)

            print("{: 6d} {:5.3f} {:5.3f} {:5.3f} {}".format(epoch,loss,accuracy,relevance,s))
            row = [epoch, loss, accuracy, relevance, s]
            self.rows.append(row)

# #. move into Data class?
# def create_dataset(dat, nlookback=1):
#     """
#     convert an array of values into a dataset matrix
#     eg _________
#     from http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#     """
#     dataX, dataY = [], []
#     for i in range(len(dat) - nlookback):
#         a = dat[i:(i + nlookback)]
#         dataX.append(a)
#         dataY.append(dat[i + nlookback])
#     return np.array(dataX), np.array(dataY)

#. move to util
def cutoff(p):
    """
    convert probabilities to hard 0's and 1's (1 for highest probability)
    eg ___________
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




class RnnKeras(Model):
    """
    An RNN implemented with Keras - can be a SimpleRNN, LSTM, or GRU.
    """

    #. pass rnn type - SimpleRNN, LSTM, GRU
    def __init__(self, data, train_amount=1.0, n=3, k=3, nvocab=1000, nhidden=100, nepochs=10, name_includes=[]):
        """
        Create an RNN model
        data          - a Data object - source of training and testing data
        train_amount  - percent or number of training characters to use
        nvocab        - max number of vocabulary words to learn
        nhidden       - number of units in the hidden layer
        nepochs       - number of times to run through training data
        name_includes - list of properties to include in model name, eg ['nhidden']
        """
        # arguments
        self.data = data
        self.train_amount = train_amount
        self.n = n # n-1 is amount of context given to model, ie number of words used for prediction.
        self.k = k
        self.nvocab = nvocab
        self.nhidden = nhidden
        self.nepochs = nepochs

        # calculated values
        if not 'n' in name_includes:
            name_includes.append('n')
        self.name = "RNN-" + '-'.join([key+'-'+str(self.__dict__[key]) for key in name_includes]) # eg 'RNN-nhidden-10'
        self.filetitle = '%s/rnn-(n-%d-train_amount-%s-nvocab-%d-nhidden-%d-nepochs-%d)' \
                         % (data.model_folder, self.n, str(train_amount), nvocab, nhidden, nepochs)
        self.filename = self.filetitle + '.pickle'
        self.filename_h5 = self.filetitle + '.h5'
        self.trained = False
        self.load_time = None
        self.save_time = None
        self.train_time = None
        self.test_time = None

        # create the keras model
        print("Create model " + self.name)
        self.rnn = Sequential()
        #. change rnn type here - SimpleRNN, LSTM, or GRU
        self.rnn.add(SimpleRNN(self.nhidden, input_dim=self.nvocab))
        self.rnn.add(Dense(self.nvocab)) #. this isn't part of the RNN already?
        self.rnn.add(Activation('softmax'))
        # note: can't make a custom metric for relevance here because callback just passes y_true and y_pred, not probs.
        # categorical_crossentropy is faster than mean_squared_error.
        #. try different optimizers
        self.rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, force_training=False):
        """
        Train the rnn and save it, or load from file if available.
        force_training - set True to retrain rnn (ie don't load from file)
        """

        if force_training==False and os.path.isfile(self.filename):
            self.load() # see model.py - will set self.load_time
        else:

            self.batch_size = 1
            # self.batch_size = 25  #. much slower convergence than 1 - why? what does this do?

            # self.epoch_interval = 10
            self.epoch_interval = 1

            print("Training model %s on %s percent/chars of training data..." % (self.name, str(self.train_amount)))

            print("Getting training tokens...")
            with benchmark("Prepared training data"):
                tokens = self.data.tokens('train', self.train_amount) # eg ['the','dog','barked',...]
                self.vocab = vocab.Vocab(tokens, self.nvocab)
                itokens = self.vocab.get_itokens(tokens) # eg ________
                onehot = keras.utils.np_utils.to_categorical(itokens, self.nvocab) # one-hot encoding, eg _______
                #. these would be huge arrays - simpler way?
                x, y = self.vocab.create_dataset(onehot, self.n-1) # n-1 = amount of lookback / context, eg _________

            print("Starting gradient descent...")
            with benchmark("Gradient descent finished") as b:
                #. add early stopping
                # early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min') # stop if no improvement for 10 epochs
                show_loss = ShowLoss(self, x, y, self.epoch_interval) #. pass in validation dataset here, not x y
                # self.rnn.fit(x, y, nb_epoch=self.nepochs, batch_size=self.batch_size, verbose=0, callbacks=[ShowLoss(self.epoch_interval)])
                # self.rnn.fit(x, y, nb_epoch=self.nepochs, batch_size=self.batch_size, verbose=0)
                print("{:6} {:5} {:5} {:5} {}".format('epoch','loss','acc','relev','s'))
                history = self.rnn.fit(x, y, nb_epoch=self.nepochs, batch_size=self.batch_size, verbose=0,
                                       callbacks=[show_loss])
                #. what is history obj?
                # print(history)

            self.train_time = b.time
            self.trained = True
            self.train_results = pd.DataFrame(show_loss.rows, columns=['Epoch','Loss','Accuracy','Relevance','Prediction'])

            # save the model
            self.save()

    def save(self):
        """
        Save the model to file - overrides base model class method.
        """
        Model.save(self, exclude=['rnn'])
        self.rnn.save(self.filename_h5)

    def load(self):
        """
        Load the model from file - overrides base model class method.
        """
        Model.load(self)
        self.rnn = load_model(self.filename_h5)

    #. ?
    def get_prediction(self, x):
        """
        predict the next word in the sequence
        x is onehot, eg ____________?
        """
        yprobs = self.rnn.predict(x) # softmax probabilities, eg ________
        # print(yprobs)
        yonehot = cutoff(yprobs) # onehot encodings, eg ________
        # print(yonehot)
        yiwords = [row.argmax() for row in yonehot] # iword values, eg ________
        # print(yiwords)
        ywords = self.vocab.get_tokens(yiwords) # eg ___________
        # print(ywords)
        s = ' '.join(ywords) # eg _____________
        # print(s)
        return s

    #. refactor!
    def predict(self, prompt):
        """
        Get the k most likely next tokens following the given string.
        eg model.predict('The cat') -> [('slept',0.12), ('barked',0.08), ('meowed',0.07)]
        """
        x, y = self.vocab.prompt_to_onehot(prompt, self.n) # eg ___________
        probs = self.rnn.predict_proba(x) # eg __________
        best_words = self.vocab.probs_to_word_probs(probs, self.k) # eg ____________
        return best_words


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
    model = RnnKeras(data, n=3, nvocab=30, nhidden=6, nepochs=50)
    # model.save()
    # data = Data('animals')
    # model = RnnKeras(data, n=3, nvocab=30, nhidden=6, nepochs=25, train_amount=1000)
    # model = RnnKeras(data, n=4, nvocab=30, nhidden=6, nepochs=50, train_amount=1000)
    # model = RnnKeras(data, n=5, nvocab=30, nhidden=6, nepochs=50, train_amount=1000)
    # data = Data('gutenbergs')

    # model.train() # load model from file if available
    model.train(force_training=True)

    # print(util.table(model.train_results))
    # print()
    # print(model.rnn.get_weights())
    # print()
    # model.save()
    # model.load()


    # predict next word after a prompt
    # s = 'The cat'
    prompt = 'a b c'
    word_probs = model.predict(prompt)
    print('prediction')
    print(prompt)
    print(word_probs) # 'd' should be first in list


    # works
    # # plot training curves
    # model.train_results.plot(x='Epoch',y=['Loss','Accuracy','Relevance'])
    # plt.show()

    model.test(test_amount=100)
    print('relevance',model.test_score)
    print(util.table(model.test_samples))
    print()

