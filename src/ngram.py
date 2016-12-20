
"""
n-gram word prediction model

"""

from __future__ import print_function, division

import nltk
from nltk import tokenize

# import pickle
import cPickle as pickle # faster version of pickle
from pprint import pprint, pformat


class Ngram():
    """
    n-gram model - initialize with n.
    Stores a sparse multidimensional array of word counts.
    The array is implemented as a dict of dicts.
    """
    def __init__(self, n):
        self.n = n
        self.d = {}

    def train(self, s):
        """
        Train the ngram model with the given string s.
        """
        print("tokenize words")
        #. can we feed this a generator instead?
        words = tokenize.word_tokenize(s)
        print("get ngrams")
        word_tuples = nltk.ngrams(words, self.n)
        print("add ngrams to model")
        for word_tuple in word_tuples:
            self.increment(word_tuple)

    def increment(self, word_tuple):
        """
        Increment the value of the multidimensional array at index word_tuple by 1.
        """
        nwords = len(word_tuple)
        d = self.d
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
        Get a random following word following the given sequence.
        """
        # get the last dictionary, which contains the subsequent words and their counts
        d = self.d
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


    def predict(self, words):
        """
        Get the most likely next word following the given sequence.
        """
        # get the last dictionary, which contains the subsequent words and their counts
        d = self.d
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
        return pformat(self.d) # from pprint module

    # def get_probabilities(model):
        # for word in model:
        # for i, word in enumerate(tuple):
        # d = model[word0]

    def save(self, filename):
        """
        Save ngram model to given filename.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load ngram model from given filename
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            return model

# works
# use word tokenizer
# words = word_tokenize(s)
# wlist = [words[i:] for i in range(n)]
# d = {}
# for words in zip(*wlist):
#     print words

# works
# sentences = sent_tokenize(s)
# for sentence in sentences:
#     print sentence



