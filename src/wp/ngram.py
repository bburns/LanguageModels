
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


class NgramModel(model.Model):
    """
    n-gram model - initialize with n.
    Inherits some code from Model class.

    Stores a sparse multidimensional array of token counts.
    The sparse array is implemented as a dict of dicts.
    """

    def __init__(self, n, model_folder='.', nchars=None):
        """
        Create an n-gram model.
        """
        self.n = n  # the n in n-gram
        self.nchars = nchars  # number of characters trained on #. should be ntrain_tokens?
        self._d = {} # dictionary of dictionary of ... of counts
        self.trained = False
        # self.model_folder = model_folder
        self.name = "n-gram-(nchars-%d-n-%d)" % (nchars, n)
        self.filename = model_folder + '/' + self.name + '.pickle'

    def train(self, tokens, nepochs='unused'):
        """
        Train the ngram model with the given token stream.
        """
        print("get ngrams, n=%d" % self.n)
        #. is this a generator? will want n=10+ for rnn. if not make one
        token_tuples = nltk.ngrams(tokens, self.n)
        print("add ngrams to model")
        for token_tuple in token_tuples:
            self._increment_count(token_tuple)
        self.trained = True

    def _increment_count(self, token_tuple):
        """
        Increment the value of the multidimensional array at given index (token_tuple) by 1.
        """
        ntokens = len(token_tuple)
        d = self._d
        # need to iterate down the token stream to find the last dictionary,
        # where you can increment the counter.
        for i, token in enumerate(token_tuple):
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

    def generate_token(self, tokens):
        """
        Get a random token following the given sequence.
        """
        if self.n==1:
            tokens = [] # no context - will just return a random token from vocabulary
        else:
            tokens = tokens[-self.n+1:] # an n-gram can only see the last n tokens
        # print(tokens)
        # get the final dictionary, which contains the subsequent tokens and their counts
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

    # def generate(self, k=1):
    def generate(self):
        """
        # Generate k sentences of random text.
        Generate sentence of random text.
        """
        start1 = 'END'
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
        # for i in range(k):
        while True:
            next = self.generate_token(input)
            input.pop(0)
            input.append(next)
            output.append(next)
            if next=='END':
                break
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


if __name__ == '__main__':

    import itertools
    import numpy as np

    # unknown_token = "UNKNOWN"
    # sentence_end_token = "END"

    # s = "The dog barked. The cat meowed. The dog ran away. The cat slept."
    # print(s)

    # # nvocab = 10

    # # split text into sentences
    # sentences = nltk.sent_tokenize(s)
    # print(sentences)

    # # append END tokens
    # sentences = ["%s %s" % (sent, sentence_end_token) for sent in sentences]
    # print(sentences)

    # # Tokenize the sentences into words
    # tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # print(tokenized_sentences)

    # # Count the word frequencies
    # word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    # vocab = word_freq.most_common(nvocab-1)
    # print('most common words',vocab)
    # index_to_word = [pair[0] for pair in vocab]
    # index_to_word.append(unknown_token)
    # print('index to word',index_to_word)
    # word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    # print('word to index',word_to_index)

    # # Replace all words not in our vocabulary with the unknown token
    # print('replace unknown words with UNKNOWN token')
    # for i, sent in enumerate(tokenized_sentences):
    #     tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    # print(tokenized_sentences)

    # # Create the training data
    # print('Create training data:')
    # X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    # y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    # # print('X_train:',X_train[500])
    # # print('y_train:',y_train[500])
    # print('X_train:',X_train) # eg tokenized: The dog ran down the hill . END
    # print('y_train:',y_train) # eg tokenized: dog ran down the hill . END


    # --------------------------------------

    strain = "the dog barked . END the cat meowed . END the dog ran away . END the cat slept ."
    stest = "the dog"
    # stest = "the dog slept"
    train_tokens = strain.split()
    test_tokens = stest.split()

    model = NgramModel(n=1)
    model.train(train_tokens)
    token = model.generate_token(test_tokens)
    print(test_tokens)
    print('prediction:',token)
    tokens = model.generate()
    print(' '.join(tokens))
    print()

    model = NgramModel(n=2)
    model.train(train_tokens)
    token = model.generate_token(test_tokens)
    print(test_tokens)
    print('prediction:',token)
    tokens = model.generate()
    print(' '.join(tokens))
    print()



    # print('predictions')
    # predictions = model.predict(X_train[1], 3)
    # # print(predictions.shape)
    # print(predictions)
    # # s = [index_to_word[i] for i in predictions]
    # # print(s)
    # print('actual')
    # print(y_train[1])
    # s = [index_to_word[i] for i in y_train[1]]
    # print(s)


