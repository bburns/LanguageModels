
"""
Base Model class for use by n-gram and RNN classes.
"""

from __future__ import print_function, division
import os
import os.path
import cPickle as pickle # faster version of pickle

from benchmark import benchmark
import util


class Model(object):
    """
    Base model for n-gram and RNN classes.
    """

    def save(self):
        """
        Save the model to the default filename.
        """
        #. time this? but can't save it with the object
        folder = os.path.dirname(self.filename)
        util.mkdir(folder)
        with benchmark("Save model " + self.name) as b:
            with open(self.filename, 'wb') as f:
                # pickle.dump(self, f)
                pickle.dump(self.__dict__, f, 2)
        # self.save_time = b.time
        # return self.save_time

    def load(self):
        """
        Load model from the default filename.
        """
        if os.path.isfile(self.filename):
            with benchmark("Load model " + self.name) as b:
                with open(self.filename, 'rb') as f:
                    d = pickle.load(f)
                    self.__dict__.update(d)
            self.load_time = b.time
        else:
            self.load_time = None
        return self.load_time


    def test(self, k=3, test_amount=1.0):
        """
        Test the model and return the accuracy score.
        k - number of words to predict
        test_amount - amount of test data to use, in percent or nchars
        """
        # get the test tokens
        tokens = self.data.tokens('test', test_amount)
        ntokens = len(tokens)
        # run test on the models
        nright = 0
        with benchmark("Test model " + self.name) as b: # time it
            npredictions = ntokens - self.n
            for i in range(npredictions): # iterate over all test tokens
                prompt = tokens[i:i+self.n-1]
                actual = tokens[i+self.n-1]
                token_probs = self.predict(prompt, k) # eg [('barked',0.031),('slept',0.025)...]
                #. add selection to samples
                print('prompt',prompt,'actual',actual,'token_probs',token_probs)
                if token_probs: # can be None
                    predicted_tokens = [token_prob[0] for token_prob in token_probs]
                    if actual in predicted_tokens:
                        nright += 1
            accuracy = nright / npredictions if npredictions>0 else 0
        self.test_time = b.time
        self.test_score = accuracy
        return accuracy

    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepochs=100, evaluate_loss_after=5):
        """
        Train model with Stochastic Gradient Descent (SGD)
        X_train             - the training data set
        y_train             - the training data labels
        learning_rate       - initial learning rate for SGD
        nepochs             - number of times to iterate through the complete dataset
        evaluate_loss_after - evaluate the loss after this many epochs
        We keep track of the losses so we can plot them later
        """
        losses = []
        nexamples_seen = 0
        for nepoch in range(nepochs):
            # optionally evaluate the loss
            if (nepoch % evaluate_loss_after == 0):
                loss = self.average_loss(X_train, y_train)
                losses.append((nexamples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after nexamples_seen=%d epoch=%d: %f" % (time, nexamples_seen, nepoch, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # for each training example... (ie each sentence in his formulation)
            for i in range(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate) # take one sgd step
                nexamples_seen += 1
        return losses

