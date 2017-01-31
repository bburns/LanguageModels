
"""
n-gram word prediction model

Stores a sparse multidimensional array of token counts.
The sparse array is implemented as a dict of dicts.
"""

import os
import random
from pprint import pprint, pformat

import nltk
from nltk import tokenize
import tabulate

import util
from benchmark import benchmark


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

    def train(self, train_amount=1.0, debug=0):
        """
        Train the model
        """
        print("Training model n=%d..." % self.n)
        tokens = self.data.sequence
        tuples = nltk.ngrams(tokens, self.n)
        for tuple in tuples:
            self._increment_count(tuple) # add ngram counts to model
        if debug:
            print('d keys',list(self._d.keys())[:5])
            print('d values',list(self._d.values())[:5])

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
        probs = [[]]
        for iword in range(nvocab):
            ncount = d.get(iword)
            if ncount:
                pct = ncount/ntotal if ntotal!=0 else 0
            else:
                pct = 0.0
            probs[0].append(pct)
        return probs




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

    # def __str__(self):
    #     """
    #     Return string representation of model.
    #     """
    #     propnames = "name data n train_amount trained train_time \
    #                  load_time save_time test_time test_score filename".split()
    #     rows = []
    #     for propname in propnames:
    #         propvalue = self.__dict__[propname]
    #         row = [propname, propvalue]
    #         rows.append(row)
    #     s = tabulate.tabulate(rows, ['Property', 'Value'])
    #     # s = str(rows)
    #     # s =
    #     # s = self.name + '\n'
    #     # s +=
    #     # self.data = data # lightweight interface for data files
    #     # self.n = n  # the n in n-gram
    #     # self.train_amount = train_amount
    #     # self._d = {} # dictionary of dictionary of ... of counts
    #     # self.trained = False
    #     # self.name = "ngram (n=%d)" % n
    #     # self.filename = "%s/ngram-(n-%d-amount-%.4f).pickle" % (data.model_folder, n, train_amount)
    #     # print("Create model " + self.name)
    #     return s



if __name__ == '__main__':

    from data import Data
    data = Data('alice1')
    data.prepare(nvocab=20)

    # n=1
    # if 1:
    # for n in (1,2,3,4,5):
    for n in (1,2):
        model = Ngram(data, n=n)
        model.train(debug=1)

        util.uprint('Final epoch generated text:', util.generate_text(model, data, n))
        print()
        # model.test(test_amount=2000)
        # print('accuracy:', model.test_accuracy)
        # print('relevance:', model.test_relevance)
        # print('sample predictions:')
        # df = model.test_samples
        # print(tabulate(model.test_samples, showindex=False, headers=df.columns))
        # # print(df)
        # # print(model._d)
        # s = model.generate()
        # # print('generate:', s)
        # print('generate:', repr(s)) # weird symbols sometimes crash print
        # print()



