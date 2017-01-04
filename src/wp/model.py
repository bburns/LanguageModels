
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


    def test(self, k=3, test_amount=1.0): #. use all data by default?
        """
        Test the model and return the accuracy score.
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


