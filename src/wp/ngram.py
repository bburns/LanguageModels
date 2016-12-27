
"""
n-gram word prediction model

Basic version - no backoff or smoothing.
"""

from __future__ import print_function, division

import random
from pprint import pprint, pformat

import nltk
from nltk import tokenize

import model
import util


# class NgramModel(object):
class NgramModel(model.Model):
    """
    n-gram model - initialize with n.
    Inherits some code from Model class.

    Stores a sparse multidimensional array of token counts.
    The sparse array is implemented as a dict of dicts.
    """

    def __init__(self, n, modelfolder='.', nchars=None):
        """
        Create an n-gram model.
        """
        self.n = n  # the n in n-gram
        self.nchars = nchars  # number of characters trained on #. should be ntrain_tokens eh?
        self.modelfolder = modelfolder
        self.name = "n-gram (n=%d)" % n
        self._d = {} # dictionary of dictionary of ... of counts

    def filename(self):
        """
        Get default filename for model.
        """
        classname = type(self).__name__ # ie 'NgramModel'
        params = (('nchars',self.nchars),('n',self.n))
        sparams = util.encode_params(params) # eg 'nchars-1000-n-3'
        filename = "%s/%s-%s.pickle" % (self.modelfolder, classname, sparams)
        return filename

    def train(self, tokens):
        """
        Train the ngram model with the given tokens.
        """
        print("get ngrams, n=%d" % self.n)
        token_tuples = nltk.ngrams(tokens, self.n)
        print("add ngrams to model")
        for token_tuple in token_tuples:
            self.increment(token_tuple)

    def trained(self):
        """
        Has this model been trained yet?
        """
        if self._d:
            return True
        else:
            return False

    #. prefix with _ ?
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

    def generate(self, k):
        """
        Generate k tokens of random text.
        """
        start1 = '.'
        output = []
        input = [start1]
        if self.n>=3:
            start2 = random.choice(self._d[start1].keys())
            input.append(start2)
            output.append(start2)
        if self.n>=4:
            start3 = random.choice(self._d[start1][start2].keys())
            input.append(start3)
            output.append(start3)
        for i in range(k-1):
            next = self.get_random(input)
            input.pop(0)
            input.append(next)
            output.append(next)
        return output

    def predict(self, tokens, k):
        """
        Get the most likely next k tokens following the given sequence.
        """
        #. add assert len(tokens)==self.n, or ignore too much/not enough info?
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
        best_tokens = util.get_best_tokens(d, k)
        return best_tokens

    def __str__(self):
        """
        Return description of model.
        """
        # return pformat(self._d) # from pprint module
        s = ''
        s += self.name + '\n'
        s += 'vocab size %d\n' % len(self._d)
        return s


