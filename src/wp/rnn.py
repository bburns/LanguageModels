
"""
Recurrent neural network (RNN) model
"""

from __future__ import print_function, division

import os
import os.path
import random
import heapq
import cPickle as pickle # faster version of pickle
from pprint import pprint, pformat

import nltk
from nltk import tokenize



class RnnModel(object):
    """
    Recurrent neural network (RNN) model
    """

    def __init__(self, modelfolder='.', nchars=None):
        """
        Create an rnn model
        """
        self.modelfolder = modelfolder
        self.name = 'rnn'

    def filename(self):
        """
        Get default filename for model.
        """
        classname = type(self).__name__ # ie 'RnnModel'
        params = (('nchars',self.nchars))
        sparams = encode_params(params) # eg 'nchars-1000'
        filename = "%s/%s-%s.pickle" % (self.modelfolder, classname, sparams)
        return filename

    def train(self, tokens):
        """
        Train the rnn model with the given tokens.
        """
        # print("get ngrams, n=%d" % self.n)
        # token_tuples = nltk.ngrams(tokens, self.n)
        # print("add ngrams to model")
        # for token_tuple in token_tuples:
        #     self.increment(token_tuple)
        pass

    def trained(self):
        """
        Has this model been trained yet?
        """
        # if self._d:
        #     return True
        # else:
        #     return False
        return False

    def get_random(self, tokens):
        """
        Get a random token following the given sequence.
        """
        pass

    def generate(self, k):
        """
        Generate k tokens of random text.
        """
        pass

    def predict(self, tokens, k):
        """
        Get the most likely next k tokens following the given sequence.
        """
        pass

    def __str__(self):
        """
        Return model as a string.
        """
        pass

    #. move save/load to baseclass wp.Model
    def save(self, filename=None):
        """
        Save the model to the default or given filename.
        """
        if filename is None:
            filename = self.filename()
        try:
            folder = os.path.dirname(filename)
            os.mkdir(folder)
        except:
            pass
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename=None):
        """
        Load model from the given or default filename.
        """
        if filename is None:
            filename = self.filename()
        if os.path.isfile(filename):
            print("load model")
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                return model
        else:
            return self





