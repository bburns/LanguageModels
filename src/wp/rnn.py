
"""
Recurrent neural network (RNN) model
"""

from __future__ import print_function, division

import nltk
from nltk import tokenize

import cPickle as pickle # faster version of pickle
from pprint import pprint, pformat


class RnnModel():
    """
    Recurrent neural network (RNN) model
    """

    def __init__(self):
        """
        Create an rnn model
        """
        pass

    def train(self, s):
        """
        Train the rnn model with the given string s.
        """
        pass

    def predict(self, words):
        """
        Get the most likely next k words following the given sequence.
        """
        pass

    def save(self, filename):
        """
        Save the model to the given filename.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a model from the given filename.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            return model




