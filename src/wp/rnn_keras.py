
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



class ShowLoss(keras.callbacks.Callback):
    """
    A callback class to show loss, accuracy, and prediction during training.
    """
    #. pass in a pandas table and col formats, print last row each step using tabulate?
    def __init__(self, m, x_validate, y_validate, k, epoch_interval=10):
        self.m = m
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.k = k
        self.epoch_interval = epoch_interval
        self.rows = []

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.epoch_interval == 0:
            loss, accuracy = self.model.evaluate(self.x_validate, self.y_validate, verbose=0)
            predictions = self.m.get_predictions(self.x_validate)
            # relevance = util.get_relevance(self.y_validate, yprobs) #. pass k
            relevance = util.get_relevance(self.y_validate, yprobs, k)
            row = [epoch, loss, accuracy, relevance, predictions]
            print("{: 6d} {:5.3f} {:5.3f} {:5.3f} {}".format(*row))
            self.rows.append(row)


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
        self.vocab = None

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

            print("Training model %s on %s of training data..." % (self.name, str(self.train_amount)))

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
                show_loss = ShowLoss(self, x, y, self.k, self.epoch_interval) #. pass in validation dataset here, not x y
                print("{:6} {:5} {:5} {:5} {}".format('epoch','loss','acc','relev','predictions'))
                history = self.rnn.fit(x, y, nb_epoch=self.nepochs, batch_size=self.batch_size, verbose=0, callbacks=[show_loss])
                #. what is history obj?
                # print(history)

            self.train_time = b.time
            self.trained = True
            self.train_results = pd.DataFrame(show_loss.rows, columns=['Epoch','Loss','Accuracy','Relevance','Predictions'])

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

    def get_predictions(self, xonehots):
        """
        predict the next words in the given sequence.
        called by ShowLoss callback class.
        xonehots eg ____________
        """
        yprobs = self.rnn.predict(xonehots) # softmax probabilities, eg ______
        yonehots = util.cutoff(yprobs) # onehot encodings, eg ______
        yiwords = [row.argmax() for row in yonehots] # iword values, eg ______
        ywords = self.vocab.get_tokens(yiwords) # eg _______
        s = ' '.join(ywords) # eg ________
        return s


    #. refactor!
    def predict(self, prompt):
        """
        Get the k most likely next tokens following the given string.
        eg model.predict('The cat') -> [('slept',0.12), ('barked',0.08), ('meowed',0.07)]
        """
        x, y = self.vocab.prompt_to_onehot(prompt, self.n) # eg ___________
        probs = self.rnn.predict_proba(x, verbose=0) # eg __________
        best_words = self.vocab.probs_to_word_probs(probs, self.k) # eg ____________
        return best_words


if __name__=='__main__':

    import matplotlib.pyplot as plt
    from data import Data

    np.random.seed(0)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # select dataset and build model

    # data = Data('abcd') # vocab is 7 tokens: . END UNKNOWN a b c d
    # model = RnnKeras(data, n=3, nvocab=7, nhidden=4, nepochs=200) # works
    # prompt = 'a b c'

    # data = Data('alphabet')
    # nepochs vs nhidden - cramming info into smaller hidden layer takes more work
    # model = RnnKeras(data, n=3, nvocab=30, nhidden=30, nepochs=20) # works
    # model = RnnKeras(data, n=3, nvocab=30, nhidden=15, nepochs=40) # works
    # model = RnnKeras(data, n=3, nvocab=30, nhidden=10, nepochs=40) # works
    # model = RnnKeras(data, n=3, nvocab=30, nhidden=5, nepochs=800) # nearly works
    # nepochs vs n - increasing context doesn't require much work
    # model = RnnKeras(data, n=3, nvocab=30, nhidden=15, nepochs=40) # works
    # model = RnnKeras(data, n=4, nvocab=30, nhidden=15, nepochs=40) # works
    # model = RnnKeras(data, n=5, nvocab=30, nhidden=15, nepochs=40) # works
    # model = RnnKeras(data, n=6, nvocab=30, nhidden=15, nepochs=50) # works
    # model = RnnKeras(data, n=7, nvocab=30, nhidden=15, nepochs=50) # works
    # model = RnnKeras(data, n=8, nvocab=30, nhidden=15, nepochs=50) # works
    # model = RnnKeras(data, n=9, nvocab=30, nhidden=15, nepochs=50) # works
    # model = RnnKeras(data, n=10, nvocab=30, nhidden=15, nepochs=60) # works
    # model = RnnKeras(data, n=20, nvocab=30, nhidden=15, nepochs=60) # works
    # prompt = 'a b c d e f g h i j k l m'

    # data = Data('animals')
    # model = RnnKeras(data, n=5, nvocab=60, nhidden=30, nepochs=100) # nearly works
    # prompt = 'the cat and dog'

    data = Data('alice1')
    # model = RnnKeras(data, n=4, nvocab=100, nhidden=50, nepochs=10) # eh
    # model = RnnKeras(data, n=5, nvocab=200, nhidden=100, nepochs=10)
    model = RnnKeras(data, n=5, nvocab=200, nhidden=100, nepochs=1)
    prompt = 'the white rabbit said'

    # train model
    # model.train(force_training=True)
    model.train()
    print(util.table(model.train_results))
    print()

    # # show vocab
    # print('vocab',model.vocab)

    # # show weight matrices
    # weights = model.rnn.get_weights()
    # U, W, b, V, c = weights
    # print('U')
    # print(U)
    # print('W')
    # print(W)
    # print('b')
    # print(b)
    # print()
    # print('V')
    # print(V)
    # print('c')
    # print(c)
    # print()

    # predict next word after a prompt (define above)
    word_probs = model.predict(prompt)
    print('prediction')
    print(prompt)
    print(word_probs) # 'd' should be first in list
    print()

    # test the model against the test dataset
    model.test(test_amount=1000)
    print('relevance',model.test_score)
    print()

    # show sample predictions
    print('sample predictions')
    print(util.table(model.test_samples))
    print()


    # data = Data('animals')
    # model = RnnKeras(data, n=3, nvocab=30, nhidden=6, nepochs=25, train_amount=1000)
    # model = RnnKeras(data, n=4, nvocab=30, nhidden=6, nepochs=50, train_amount=1000)
    # model = RnnKeras(data, n=5, nvocab=30, nhidden=6, nepochs=50, train_amount=1000)
    # data = Data('gutenbergs')

    # model.train() # load model from file if available
    # model.train(force_training=True)

    # print(util.table(model.train_results))
    # print()
    # print(model.rnn.get_weights())
    # print()

    # # predict next word after a prompt
    # # prompt = 'The cat'
    # # prompt = 'The white rabbit'
    # word_probs = model.predict(prompt)
    # print('prediction')
    # print(prompt)
    # print(word_probs)

    # works
    # # plot training curves
    # model.train_results.plot(x='Epoch',y=['Loss','Accuracy','Relevance'])
    # plt.show()

    # model.test(test_amount=100)
    # print('relevance',model.test_score)
    # print(util.table(model.test_samples))
    # print()

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

