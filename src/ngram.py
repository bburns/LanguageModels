
"""
N-gram word prediction model
Stores a sparse multidimensional array (dict of dict) of token counts.
"""


# --------------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------------


import os
import random
from pprint import pprint, pformat

import numpy as np
import pandas as pd
import nltk
from nltk import tokenize
from tabulate import tabulate

import util
from benchmark import benchmark


#. this class could be in a separate file, ngram_model.py, to be more like rnn.py
# --------------------------------------------------------------------------------
# define ngram class
# --------------------------------------------------------------------------------

class Ngram():

    def __init__(self, data, n=3):
        """
        Create an n-gram model.
        data - a Data object - source of data
        n - the n in n-gram
        """
        self.data = data
        self.n = n
        self._d = {} # dictionary of dictionary of ... of counts

    def fit(self, train_amount=1.0, debug=0):
        """
        Train the model
        """
        print("Training model n=%d..." % self.n)
        tokens = self.data.sequence
        tuples = nltk.ngrams(tokens, self.n)
        for tuple in tuples:
            self._increment_count(tuple) # add ngram counts to model
        # if debug:
            # print('d keys',list(self._d.keys())[:5])
            # print('d values',list(self._d.values())[:5])

    def _increment_count(self, tuple):
        """
        Increment the value of the multidimensional array at given index (token tuple) by 1.
        """
        ntokens = len(tuple)
        d = self._d
        # need to iterate down the token stream to find the last dictionary,
        # where you can increment the counter.
        for i, token in enumerate(tuple):
            if i==ntokens-1: # at last dictionary
                if token in d:
                    d[token] += 1
                else:
                    d[token] = 1
            else:
                if token in d:
                    d = d[token]
                else:
                    d[token] = {}
                    d = d[token]


    def predict_proba(self, x, verbose=0):
        """
        predict probability of following words, over entire vocabulary
        this method follows keras's api (ie duck typing)
        """
        # get the last dictionary, which contains the subsequent tokens and their counts
        d = self._d
        tokens = x[0]
        for token in tokens:
            if token in d:
                d = d[token]
            else:
                d = {}
        nvocab = self.data.nvocab
        ntotal = sum(d.values()) # total occurrences of subsequent tokens
        # probs = [[]]
        probs = np.zeros((1,nvocab))
        for iword in range(nvocab):
            ncount = d.get(iword)
            if ncount:
                pct = ncount/ntotal if ntotal!=0 else 0
            else:
                pct = 0.0
            # probs[0].append(pct)
            probs[0,iword] = pct
        return probs


    def test(self, nsamples=10):
        """
        Test the model and return the accuracy, relevance score and some sample predictions.
        nsamples - number of sample predictions to record in self.test_samples
        Returns nothing, but sets
            self.test_time
            self.test_samples
            self.test_accuracy
            self.test_relevance
        and saves the model with those values to a file.
        """
        tokens = self.data.sequence
        ntokens = len(tokens)
        npredictions = ntokens - self.n #.
        nsample_spacing = max(int(ntokens / nsamples), 1)
        samples = []
        naccurate = 0
        nrelevant = 0
        sample_columns = ['Prompt','Predictions','Actual','Status']
        print("Testing model n=%d..." % self.n)
        for i in range(npredictions): # iterate over all test tokens
            #. refactor
            prompt = tokens[i:i+self.n-1]
            actual = tokens[i+self.n-1]
            probs = self.predict_proba([prompt])
            iword_probs = util.get_best_iword_probs(probs, k=3)
            accurate = False
            if iword_probs: # can be None
                predicted_tokens = [token_prob[0] for token_prob in iword_probs]
                accurate = (actual == predicted_tokens[0])
                if accurate:
                    naccurate += 1
                relevant = (actual in predicted_tokens)
                if relevant:
                    nrelevant += 1
            # add sample predictions
            if (i % nsample_spacing) == 0:
                sprompt = ' '.join([self.data.iword_to_word[iword] for iword in prompt])
                sactual = self.data.iword_to_word[actual]
                #. refactor
                # word = token_prob[0]
                # pct = token_prob[1]*100
                if iword_probs:
                    spredictions = '  '.join(['%s (%.1f%%)' % \
                                              # (iword_prob[0], iword_prob[1]*100) \
                                              (self.data.iword_to_word[iword_prob[0]], iword_prob[1]*100) \
                                              for iword_prob in iword_probs])
                else:
                    spredictions = '(none)'
                saccurate = 'OK' if accurate else 'FAIL'
                sample = [sprompt, spredictions, sactual, saccurate]
                samples.append(sample)
        relevance = nrelevant / npredictions if npredictions>0 else 0
        accuracy = naccurate / npredictions if npredictions>0 else 0
        # self.test_time = b.time
        self.test_accuracy = accuracy
        self.test_relevance = relevance
        self.test_samples = pd.DataFrame(samples, columns=sample_columns)
        # self.save() # save test time, score, samples

    # def generate_token(self, tokens):
    #     """
    #     Get a random token following the given sequence.
    #     """
    #     if self.n==1:
    #         tokens = [] # no context - will just return a random token from vocabulary
    #     else:
    #         tokens = tokens[-self.n+1:] # an n-gram can only see the last n tokens
    #     # get the final dictionary, which contains the subsequent tokens and their counts
    #     d = self._d
    #     for token in tokens:
    #         if token in d:
    #             d = d[token]
    #         else:
    #             return None
    #     # pick a random token according to the distribution of subsequent tokens
    #     ntotal = sum(d.values()) # total occurrences of subsequent tokens
    #     p = random.random() # a random value 0.0-1.0
    #     stopat = p * ntotal # we'll get the cumulative sum and stop when we get here
    #     ntotal = 0
    #     for token in d.keys():
    #         ntotal += d[token]
    #         if stopat < ntotal:
    #             return token
    #     return d.keys()[-1] # right? #. test

    # def generate(self):
    #     """
    #     Generate sentence of random text.
    #     """
    #     start1 = '.' #. magic
    #     output = []
    #     input = [start1]
    #     if self.n>=3:
    #         start2 = random.choice(list(self._d[start1].keys()))
    #         input.append(start2)
    #         output.append(start2)
    #     if self.n>=4:
    #         start3 = random.choice(list(self._d[start1][start2].keys()))
    #         input.append(start3)
    #         output.append(start3)
    #     if self.n>=5:
    #         start4 = random.choice(list(self._d[start1][start2][start3].keys()))
    #         input.append(start4)
    #         output.append(start4)
    #     while True:
    #         next = self.generate_token(input)
    #         input.pop(0)
    #         input.append(next)
    #         output.append(next)
    #         if next=='.': #. magic
    #             break
    #     sentence = ' '.join(output)
    #     return sentence

    # # def predict(self, tokens):
    # def predict(self, prompt):
    #     """
    #     Get the k most likely subsequent tokens following the given string.
    #     """
    #     #. use Vocab class?
    #     s = prompt.lower()
    #     tokens = prompt.split()
    #     #. add assert len(tokens)==self.n, or ignore too much/not enough info?
    #     # get the last dictionary, which contains the subsequent tokens and their counts
    #     d = self._d
    #     for token in tokens:
    #         if token in d:
    #             d = d[token]
    #         else:
    #             return None
    #     # find the most likely subsequent token
    #     # see http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    #     # maxtoken = max(d, key=d.get)
    #     # return maxtoken
    #     best_tokens = util.get_best_tokens(d, self.k)
    #     return best_tokens



if __name__ == '__main__':

    from data import Data
    data = Data('alice1')
    # data = Data('gutenbergs')
    data.prepare(nvocab=10000)

    for n in (1,2,3,4,5):
        model = Ngram(data, n=n)

        #. pass validation%
        model.fit(debug=1)

        util.uprint(util.generate_text(model, data, n))
        print()

        #. need to report validation accuracy here, not test accuracy!
        model.test()
        print('accuracy:', model.test_accuracy)
        print('relevance:', model.test_relevance)

        # print('sample predictions:')
        # df = model.test_samples
        # print(tabulate(model.test_samples, showindex=False, headers=df.columns))

        # print('generated text:')
        # nsentences = 10
        # nwords_to_generate = 20
        # k = 10
        # for i in range(nsentences):
        #     util.uprint(model.generate(nwords_to_generate, k)) # weird symbols can crash print


