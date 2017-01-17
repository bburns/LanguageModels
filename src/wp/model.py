
"""
Base Model class for use by n-gram and RNN classes.
"""

import os
import os.path
from datetime import datetime
import sys
import pickle

import pandas as pd

from benchmark import benchmark
import util


class Model(object):
    """
    Base model for n-gram and RNN classes.
    """

    def save(self, exclude=[]):
        """
        Save the model to the default filename, excluding any specified attributes.
        """
        folder = os.path.dirname(self.filename)
        util.mkdir(folder)
        with benchmark("Save model " + self.name) as b:
            with open(self.filename, 'wb') as f:
                d = {k:v for (k,v) in self.__dict__.items() if not k in exclude}
                pickle.dump(d, f) # default protocol is 3
        #. time this? but can't save it with the object
        # self.save_time = b.time
        # return self.save_time

    def load(self):
        """
        Load model from the default filename.
        """
        if os.path.isfile(self.filename):
            print("Loading model " + self.name + "...")
            with benchmark("Loaded model " + self.name) as b:
                with open(self.filename, 'rb') as f:
                    d = pickle.load(f)
                    self.__dict__.update(d)
            self.load_time = b.time
        else:
            self.load_time = None
        return self.load_time

    #. add mini_batch_size - not stochastic without it?
    # or is this just minibatch size of 1? hmm, i guess so
    #. adapted from ______________
    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepochs=100, evaluate_loss_after=5):
        """
        Train model with Stochastic Gradient Descent (SGD) and return losses table.
        X_train             - the training data set
        y_train             - the training data labels
        learning_rate       - initial learning rate for SGD
        nepochs             - number of times to iterate through the complete dataset
        evaluate_loss_after - evaluate the loss after this many epochs
        """
        losses = []
        nexamples_seen = 0
        #. define col widths, print header and rows with them
        #. can do with tabulate?
        loss_columns = ['Time','Epoch','Examples Seen','Learning Rate','Loss']
        # print(*loss_columns)
        print(' | '.join(loss_columns))
        for nepoch in range(nepochs):
            # optionally evaluate the loss
            if (nepoch % evaluate_loss_after == 0):
                stime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                loss = self.average_loss(X_train, y_train)
                row = [stime, nepoch, nexamples_seen, learning_rate, loss]
                losses.append(row)
                # print(*row)
                print(' | '.join([str(val) for val in row]))
                # sys.stdout.flush() # why?
                # adjust the learning rate if loss increases (ie overshot the minimum, so slow down)
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
            # for i in range(len(y_train)):
                # self.sgd_step(X_train[i], y_train[i], learning_rate) # take one sgd step
            nsequences = len(y_train)
            # iterate over training sequences
            for i in range(nsequences):
                x_sequence = X_train[i]
                y_sequence = y_train[i]
                # this will calculate the gradient for the loss function and adjust the parameters a small amount.
                # it calls out to the subclass which should implement this method - eg see rnn.py.
                self.sgd_step(x_sequence, y_sequence, learning_rate) # take one sgd step
                nexamples_seen += 1
        df_losses = pd.DataFrame(losses, columns=loss_columns)
        return df_losses

    #. refactor
    def test(self, test_amount=1.0):
        """
        Test the model and return the accuracy score.
        test_amount - amount of test data to use, in percent or nchars
        """
        nsamples = 10 #. ?
        # get the test tokens
        tokens = self.data.tokens('test', test_amount)
        ntokens = len(tokens)
        npredictions = ntokens - self.n #..
        nsample_spacing = max(int(ntokens / nsamples), 1)
        samples = []
        # run test on the models
        nright = 0
        print("Testing model " + self.name + "...")
        sample_columns = ['Prompt','Predictions','Actual','Status']
        # print(*sample_columns)
        with benchmark("Tested model " + self.name) as b: # time it
            for i in range(npredictions): # iterate over all test tokens
                prompt = tokens[i:i+self.n-1] #..
                actual = tokens[i+self.n-1] #..
                sprompt = ' '.join(prompt) if prompt else '(none)'
                # token_probs = self.predict(prompt) # eg [('barked',0.031),('slept',0.025)...]
                token_probs = self.predict(sprompt) # eg [('barked',0.031),('slept',0.025)...]
                passed = False
                if token_probs: # can be None
                    predicted_tokens = [token_prob[0] for token_prob in token_probs]
                    passed = (actual in predicted_tokens)
                    if passed:
                        nright += 1
                # add predictions to samples
                if (i % nsample_spacing) == 0:
                    # sprompt = ' '.join(prompt) if prompt else '(none)'
                    spredictions = '  '.join(['%s (%.2f%%)' % (token_prob[0], token_prob[1]*100) \
                                              for token_prob in token_probs]) if token_probs else '(none)'
                    spassed = 'OK' if passed else 'FAIL'
                    sample = [sprompt, spredictions, actual, spassed]
                    # print(*sample)
                    samples.append(sample)
            relevance = nright / npredictions if npredictions>0 else 0
        self.test_time = b.time
        self.test_score = relevance
        self.test_samples = pd.DataFrame(samples, columns=sample_columns)
        self.save() # save test time, score, samples

