
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
    from keras.metrics import top_k_categorical_accuracy

    from model import Model
    import util
    import vocab



class RnnKeras(Model):
    """
    An RNN implemented with Keras - can be a SimpleRNN, LSTM, or GRU.
    """

    #. pass rnn type - SimpleRNN, LSTM, GRU
    # def __init__(self, data, train_amount=1.0, n=3, k=3, nvocab=1000, nhidden=100, nepochs=10, name_includes=[]):
    # def __init__(self, data, train_amount=1.0, n=3, k=3, nvocab=100, nhidden=25, nepochs=10, name_includes=[]):
    def __init__(self, data, train_amount=1.0, n=3, k=3, nvocab=100, nhidden=25, nepochs=10, rnn_type='Simple', name_includes=[]):
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
        self.rnn_type = rnn_type

        # calculated values
        # if not 'n' in name_includes:
            # name_includes.append('n')
        name_includes = ['train_amount','n','k','nvocab','nhidden','nepochs','rnn_type']
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

        rnn_classes = {
            'Simple': SimpleRNN,
            'LSTM': LSTM,
            'GRU': GRU,
        }
        rnn_class = rnn_classes[self.rnn_type]

        # create the keras model
        print("Create model " + self.name)
        self.rnn = Sequential()
        self.rnn.add(rnn_class(self.nhidden, input_dim=self.nvocab)) # this is a SimpleRNN, LSTM, or GRU
        self.rnn.add(Dense(self.nvocab)) #. this isn't part of the RNN already?
        self.rnn.add(Activation('softmax'))
        # note: categorical_crossentropy is faster than mean_squared_error.
        # self.rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','top_k_categorical_accuracy'])
        def top_3_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=3)
        metrics = ['accuracy', top_3_accuracy]
        self.rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)


    def train(self, force_training=False):
        """
        Train the rnn and save it, or load from file if available.
        force_training - set True to retrain rnn (ie don't load from file)
        Returns nothing, but sets
            self.vocab - a Vocab object containing the nvocab most used words
            self.train_results - a dataframe with Loss, accuracy, relevance vs nepoch
            self.train_time - total nsecs to train
            self.trained - sets True
        and saves the model with those values to a file.
        """
        if force_training==False and os.path.isfile(self.filename):
            self.load() # see model.py - will set self.load_time
        else:
            self.batch_size = 1
            # self.batch_size = 25  #. much slower convergence than 1 - why? what does this do?
            self.epoch_interval = max(1,int(self.nepochs/20))

            print("Training model on %s of training data..." % str(self.train_amount))

            print("Getting training and validation tokens...")
            with benchmark("Prepared training data"):
                # get training tokens
                tokens = self.data.tokens('train', self.train_amount) # eg ['the','dog','barked',...]
                self.vocab = vocab.Vocab(tokens, self.nvocab)
                itokens = self.vocab.get_itokens(tokens) # eg [1,4,3,...]
                onehot = keras.utils.np_utils.to_categorical(itokens, self.nvocab) # one-hot encoding, eg _______
                #. these would be huge arrays - simpler way?
                x, y = self.vocab.create_dataset(onehot, self.n-1) # n-1 = amount of lookback / context, eg _________

                # get validation tokens
                tokens = self.data.tokens('validate') # eg ['the','dog','barked',...]
                itokens = self.vocab.get_itokens(tokens) # eg [1,4,3,...]
                onehot = keras.utils.np_utils.to_categorical(itokens, self.nvocab) # one-hot encoding, eg _______
                #. these would be huge arrays - simpler way?
                x_validate, y_validate = self.vocab.create_dataset(onehot, self.n-1) # n-1 = amount of lookback / context, eg _________

            print("Starting gradient descent...")
            with benchmark("Gradient descent finished") as b:
                #. add early stopping
                # early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min') # stop if no improvement for 10 epochs
                columns = ['Epoch','Loss','Accuracy','Relevance','Predictions']
                fmt_header = "{:5}  {:6}  {:8}  {:9}  {}"
                fmt_rows = "{: 5d}  {:6.3f}  {:8.3f}  {:9.3f}  {}"
                print(fmt_header.format(*columns))
                show_loss = ShowLoss(self, x_validate, y_validate, self.k, fmt_rows, self.epoch_interval)
                callbacks = [show_loss]
                history = self.rnn.fit(x, y, nb_epoch=self.nepochs, batch_size=self.batch_size, callbacks=callbacks, verbose=0)
                # print(history.history) # dictionary with 'acc','top_3_accuracy','loss', but no epoch nums
                show_loss.on_epoch_begin(self.nepochs) # add final loss values to show_loss.rows

            self.train_time = b.time
            self.trained = True
            self.train_results = pd.DataFrame(show_loss.rows, columns=['Epoch','Loss','Accuracy','Relevance','Predictions'])
            self.save() # save the model

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

    #. refactor
    def get_predictions(self, yprobs):
        """
        predict the next words in the given sequence.
        called by ShowLoss callback class.
        xonehots eg ____________
        """
        yiwords = [row.argmax() for row in yprobs] # iword values, eg ______
        ywords = self.vocab.get_tokens(yiwords) # eg _______
        s = ' '.join(ywords) # eg ________
        return s

    #. refactor
    def predict(self, prompt):
        """
        Get the k most likely next tokens following the given string.
        eg model.predict('The cat') -> [('slept',0.12), ('barked',0.08), ('meowed',0.07)]
        """
        x, y = self.vocab.prompt_to_onehot(prompt, self.n) # eg ___________
        # print('x')
        # print(x)
        # print('y')
        # print(y)
        probs = self.rnn.predict_proba(x, verbose=0) # eg __________
        # print('probs')
        # print(probs)
        best_words = self.vocab.probs_to_word_probs(probs, self.k) # eg ____________
        return best_words


#. use logs to display model and calcs in tensorboard?
class ShowLoss(keras.callbacks.Callback):
    """
    A callback class to show loss, accuracy, relevance, and predictions during training.
    """
    def __init__(self, m, x_validate, y_validate, k, fmt_rows, epoch_interval=10):
        """
        m - RnnKeras model (note: the Callback class defines .model as the Keras RNN model)
        x_validate -
        y_validate -
        k -
        fmt_rows - string format for row with epoch, loss, accuracy, relevance, predictions
        epoch_interval - interval between printouts #. should specify nlinesmax, not interval
        """
        self.m = m #. get rid of this if possible - confusing
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.k = k
        self.fmt_rows = fmt_rows
        self.epoch_interval = epoch_interval
        self.rows = []

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.epoch_interval == 0:
            # loss, accuracy = self.model.evaluate(self.x_validate, self.y_validate, verbose=0)
            loss, accuracy, top_3_accuracy = self.model.evaluate(self.x_validate, self.y_validate, verbose=0)
            # yprobs = self.model.predict(self.x_validate) # softmax probabilities, eg ______
            # relevance = util.get_relevance(self.y_validate, yprobs, self.k)
            # assert relevance==top_3_accuracy # works
            relevance = top_3_accuracy
            yprobs = self.model.predict(self.x_validate[:40]) # softmax probabilities, eg ______
            predictions = self.m.get_predictions(yprobs)[:60] #. rename fn
            row = [epoch, loss, accuracy, relevance, predictions]
            print(self.fmt_rows.format(*row))
            self.rows.append(row)



if __name__=='__main__':

    import matplotlib.pyplot as plt
    plt.style.use('ggplot') # nicer style
    from data import Data

    np.random.seed(0)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # select dataset and build model

    # abcd
    data = Data('abcd') # vocab is 5 tokens: a b c d ~ (~ = unknown)
    prompt = 'a b c'
    # cram it into lower dimensions...
    # model = RnnKeras(data, n=3, nvocab=5, nhidden=4, nepochs=40) # works
    # model = RnnKeras(data, n=3, nvocab=5, nhidden=3, nepochs=150) # works
    model = RnnKeras(data, n=3, nvocab=5, nhidden=2, nepochs=300) # works
    # model = RnnKeras(data, n=3, nvocab=5, nhidden=2, nepochs=400) # works loss=0.6
    # model = RnnKeras(data, n=3, nvocab=5, nhidden=2, nepochs=800) # works loss=0.2

    # alphabet
    # data = Data('alphabet')
    # prompt = 'a b c d e f g h i j k l m'
    # nepochs vs nhidden - cramming info into smaller hidden layer takes more work
    # model = RnnKeras(data, n=3, nvocab=27, nhidden=30, nepochs=20) # works
    # model = RnnKeras(data, n=3, nvocab=27, nhidden=15, nepochs=40) # works
    # model = RnnKeras(data, n=3, nvocab=27, nhidden=10, nepochs=40) # works
    # model = RnnKeras(data, n=3, nvocab=27, nhidden=5, nepochs=800) # nearly works
    # nepochs vs n - increasing context doesn't require much work
    # model = RnnKeras(data, n=3, nvocab=27, nhidden=15, nepochs=40) # works
    # model = RnnKeras(data, n=4, nvocab=27, nhidden=15, nepochs=40) # works
    # model = RnnKeras(data, n=5, nvocab=27, nhidden=15, nepochs=40) # works
    # model = RnnKeras(data, n=6, nvocab=27, nhidden=15, nepochs=50) # works
    # model = RnnKeras(data, n=7, nvocab=27, nhidden=15, nepochs=50) # works
    # model = RnnKeras(data, n=8, nvocab=27, nhidden=15, nepochs=50) # works
    # model = RnnKeras(data, n=9, nvocab=27, nhidden=15, nepochs=50) # works
    # model = RnnKeras(data, n=10, nvocab=27, nhidden=15, nepochs=60) # works
    # model = RnnKeras(data, n=20, nvocab=27, nhidden=15, nepochs=60) # works

    # animals
    # data = Data('animals')
    # model = RnnKeras(data, n=5, nvocab=60, nhidden=30, nepochs=100) # nearly works
    # prompt = 'the cat and dog'

    # alice
    # data = Data('alice1') # has 800+ unique words
    # prompt = 'the white rabbit ran away and'
    # model = RnnKeras(data, n=4, nvocab=50, nhidden=25, nepochs=5) # fast and bad
    # model = RnnKeras(data, n=4, nvocab=100, nhidden=50, nepochs=10) # eh
    # model = RnnKeras(data, n=5, nvocab=200, nhidden=100, nepochs=10)
    # model = RnnKeras(data, n=5, nvocab=200, nhidden=100, nepochs=3)
    # model = RnnKeras(data, n=5, nvocab=400, nhidden=100, nepochs=5) # shows overfitting curve
    # rnn_type
    # model = RnnKeras(data, n=5, nvocab=400, nhidden=100, nepochs=5, rnn_type='Simple')
    # model = RnnKeras(data, n=5, nvocab=400, nhidden=100, nepochs=5, rnn_type='LSTM')
    # model = RnnKeras(data, n=5, nvocab=400, nhidden=100, nepochs=5, rnn_type='GRU')


    # train model
    model.train(force_training=True)
    # model.train()
    # print(util.table(model.train_results))
    print()

    if model.nvocab<10:
        # show vocab
        print('vocab',model.vocab)

        # # show weight matrices
        weights = model.rnn.get_weights()
        U, W, b, V, c = weights
        print('U')
        print(U)
        print('W')
        print(W)
        print('b')
        print(b)
        print()
        print('V')
        print(V)
        print('c')
        print(c)
        print()

    # predict next word after a prompt (define above)
    word_probs = model.predict(prompt)
    print('prediction')
    print(prompt)
    print(word_probs)
    print()

    # test the model against the test dataset
    model.test(test_amount=2000)
    print('relevance',model.test_score)
    print()

    # show sample predictions
    print('sample predictions')
    print(util.table(model.test_samples))
    print()

    # plot training curves
    print(util.table(model.train_results))
    model.train_results.plot(x='Epoch',y=['Loss','Accuracy','Relevance'])
    plt.title(model.name)
    plt.ylabel('Loss, Accuracy, Relevance')
    plt.show()


    #.

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

