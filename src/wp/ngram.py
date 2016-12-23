
"""
n-gram word prediction model

Basic version - no backoff or smoothing.
"""

from __future__ import print_function, division
import heapq

import nltk
from nltk import tokenize

import cPickle as pickle # faster version of pickle
from pprint import pprint, pformat




def get_best_tokens(d, k):
    """
    Return the best k tokens with their probabilities from the given dictionary.
    """
    # maxtoken = max(d, key=d.get)
    lst = list(d.items()) # eg [('a',5),('dog',1),...]
    best = heapq.nlargest(k, lst, key=lambda pair: pair[1])
    ntotal = sum(d.values())
    best_pct = [(k,v/ntotal) for k,v in best]
    return best_pct



class NgramModel():
    """
    n-gram model - initialize with n.
    Stores a sparse multidimensional array of token counts.
    The array is implemented as a dict of dicts.
    """

    def __init__(self, n):
        """
        Create an n-gram model
        """
        self.n = n  # the n in n-gram
        self.name = "n-gram (n=%d)" % n
        self._d = {} # dictionary of dictionary of ... of counts


    def train(self, tokens):
        """
        Train the ngram model with the given tokens.
        """
        print("get ngrams")
        token_tuples = nltk.ngrams(tokens, self.n)
        print("add ngrams to model")
        for token_tuple in token_tuples:
            self.increment(token_tuple)


    def increment(self, token_tuple):
        """
        Increment the value of the multidimensional array at given index (token_tuple) by 1.
        """
        ntokens = len(token_tuple)
        d = self._d
        for i, token in enumerate(token_tuple):
            if i==ntokens-1:
                if not token in d:
                    d[token] = 0
                d[token] += 1
            else:
                if not token in d:
                    d[token] = {}
                d = d[token]


    def get_random(self, tokens):
        """
        Get a random token following the given sequence.
        """
        # get the last dictionary, which contains the subsequent tokens and their counts
        d = self._d
        for token in tokens:
            if token in d:
                d = d[token]
            else:
                return None
        # pick a random token according to the distribution of subsequent tokens
        ntotal = sum(d.values()) # total occurrences of subsequent tokens
        p = random.random() # a random value 0.0-1.0
        stopat = p * ntotal # we'll get the cumulative sum and stop when we get here
        ntotal = 0
        for token in d.keys():
            ntotal += d[token]
            if stopat < ntotal:
                return token
        return d.keys()[-1] # right? #. test


    #.. this should return the top k words with their percentages
    # def predict(self, tokens):
    def predict(self, tokens, k):
        """
        Get the most likely next token following the given sequence.
        """
        # get the last dictionary, which contains the subsequent tokens and their counts
        d = self._d
        for token in tokens:
            if token in d:
                d = d[token]
            else:
                return None
        # find the most likely subsequent token
        # see http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        # maxtoken = max(d, key=d.get)
        # return maxtoken
        best_tokens = get_best_tokens(d, k)
        return best_tokens


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



