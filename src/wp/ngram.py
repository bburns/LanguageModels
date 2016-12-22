
"""
n-gram word prediction model

Basic version - no backoff or smoothing.
"""

from __future__ import print_function, division

import nltk
from nltk import tokenize

import cPickle as pickle # faster version of pickle
from pprint import pprint, pformat


class NgramModel():
    """
    n-gram model - initialize with n.
    Stores a sparse multidimensional array of word counts.
    The array is implemented as a dict of dicts.
    """

    def __init__(self, n):
        """
        Create an n-gram model
        """
        self.n = n  # the n in n-gram
        self._d = {} # dictionary of dictionary of ...

    def train(self, s):
        """
        Train the ngram model with the given string s.
        """
        print("tokenize words")
        #. can we feed this a generator instead? eg readlines?
        words = tokenize.word_tokenize(s)
        print("get ngrams")
        word_tuples = nltk.ngrams(words, self.n)
        print("add ngrams to model")
        for word_tuple in word_tuples:
            self.increment(word_tuple)

    def increment(self, word_tuple):
        """
        Increment the value of the multidimensional array at given index (word_tuple) by 1.
        """
        nwords = len(word_tuple)
        d = self._d
        for i, word in enumerate(word_tuple):
            if i==nwords-1:
                if not word in d:
                    d[word] = 0
                d[word] += 1
            else:
                if not word in d:
                    d[word] = {}
                d = d[word]

    def get_random(self, words):
        """
        Get a random word following the given sequence.
        """
        # get the last dictionary, which contains the subsequent words and their counts
        d = self._d
        for word in words:
            if word in d:
                d = d[word]
            else:
                return None
        # pick a random word according to the distribution of subsequent words
        ntotal = sum(d.values()) # total occurrences of subsequent words
        p = random.random() # a random value 0.0-1.0
        stopat = p * ntotal # we'll get the cumulative sum and stop when we get here
        ntotal = 0
        for word in d.keys():
            ntotal += d[word]
            if stopat < ntotal:
                return word
        return d.keys()[-1] # right? #. test

    #.. this should return the top k words with their percentages
    def predict(self, words):
        """
        Get the most likely next word following the given sequence.
        """
        # get the last dictionary, which contains the subsequent words and their counts
        d = self._d
        for word in words:
            if word in d:
                d = d[word]
            else:
                return None
        # find the most likely subsequent word
        # see http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        maxword = max(d, key=d.get)
        return maxword

    def __str__(self):
        """
        Return model dictionary as a string.
        """
        return pformat(self._d) # from pprint module

    # def get_probabilities(model):
        # for word in model:
        # for i, word in enumerate(tuple):
        # d = model[word0]

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



