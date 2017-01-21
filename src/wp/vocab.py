
"""
Vocab
Vocabulary class
"""

import heapq

import numpy as np
import nltk
import keras
from keras.utils.np_utils import to_categorical


class Vocab(object):
    """
    #.
    """

    def __init__(self, tokens, nvocab):
        """
        Create a vocabulary using nvocab most frequent tokens.
        Replaces others with unknown token (~).
        """

        self.nvocab = nvocab

        #. make class consts?
        self.unknown_token = '~'
        self.pad_token = '%'
        # self.end_token = '@' # not used

        # get most common words for vocabulary
        token_freqs = nltk.FreqDist(tokens)
        token_counts = token_freqs.most_common(self.nvocab-1) # eg ________
        self.index_to_token = [token_count[0] for token_count in token_counts] # eg _________
        self.index_to_token.insert(0, self.unknown_token) # so unknown_token/~/OOV will always be at index 0
        # print(self.index_to_token)
        while len(self.index_to_token) < self.nvocab:
            self.index_to_token.append(self.pad_token) # pad out the vocabulary if needed
        # self.index_to_token.sort() #. just using for alphabet dataset
        # print(self.index_to_token)
        self.token_to_index = dict([(token,i) for i,token in enumerate(self.index_to_token)]) # eg _______
        # self.nvocab = len(self.index_to_token) #? already set this? cut off with actual vocab length?
        # print(self.token_to_index)

    def __str__(self):
        return str(self.token_to_index)

    def _get_index(self, token):
        """
        Convert token to integer representation.
        """
        # tried using a defaultdict to return UNKNOWN instead of a dict, but
        # pickle wouldn't save an object with a lambda - would require defining
        # a fn just to return UNKNOWN. so this'll do.
        try:
            itoken = self.token_to_index[token]
        except:
            itoken = self.token_to_index[self.unknown_token]
        return itoken

    def get_itokens(self, tokens):
        """
        convert the string tokens into integer tokens
        eg _________
        """
        itokens = [self._get_index(token) for token in tokens]
        return itokens

    def get_tokens(self, itokens):
        """
        convert integer tokens to words
        eg _________
        """
        tokens = [self.index_to_token[itoken] for itoken in itokens]
        return tokens

    #.
    def probs_to_word_probs(self, probs, k):
        """
        convert probability distribution over vocabulary to best k words and their probabilities.
        eg ________________
        """
        next_word_probs = probs[-1]
        pairs = [(itoken,p) for itoken,p in enumerate(next_word_probs)]
        best_iwords = heapq.nlargest(k, pairs, key=lambda pair: pair[1])
        best_words = [(self.index_to_token[itoken],p) for (itoken,p) in best_iwords]
        return best_words

    #.
    def prompt_to_onehot(self, prompt, n):
        """
        this is used informally for validation
        """
        s = prompt.lower()
        #. use nltk tokenizer to handle commas, etc, or use a Vocab class
        tokens = s.split()
        tokens.append(self.unknown_token) # will be predicting this value
        itokens = self.get_itokens(tokens)
        onehot = keras.utils.np_utils.to_categorical(itokens, self.nvocab) # one-hot encoding
        x, y = self.create_dataset(onehot, n-1) # n-1 = amount of lookback / context
        return x,y

    #. refactor?
    def create_dataset(self, onehot, nlookback=1):
        """
        convert an array of values into a dataset matrix
        eg _________
        from http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
        """
        dataX, dataY = [], []
        for i in range(len(onehot) - nlookback):
            a = onehot[i:(i + nlookback)]
            dataX.append(a)
            dataY.append(onehot[i + nlookback])
        return np.array(dataX), np.array(dataY)


if __name__=='__main__':

    np.random.seed(0)

    tokens = 'the cat barked dog slept .'.split()
    nvocab = 4
    vocab = Vocab(tokens, nvocab)
    print(vocab)

    s = 'the dog barked .'
    print(s)
    tokens = s.split()
    itokens = vocab.get_itokens(tokens)
    print(itokens)

    print(vocab.index_to_token)

