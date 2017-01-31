
"""
Data module - wraps all data and handles processing.

Usage:
    data = Data('alice1')
    data.prepare(nvocab=500)
    x_train, y_train, x_validate, y_validate, x_test, y_test = data.split(n=4, nvalidate=1000, ntest=1000)
"""

import os
import os.path
import re
import random

import numpy as np
import nltk
from nltk import tokenize


class Data():

    def __init__(self, dataset):
        self.dataset = dataset
        self.folder = '../data/' + self.dataset

    def prepare(self, nvocab, seed=0, debug=False):
        """
        Prepare dataset by reading texts and tokenizing into a sequence with nvocab vocabulary words.
        """
        self.nvocab = nvocab

        print('Reading texts...') # ~1sec
        text = ''
        for filename in sorted(os.listdir(self.folder)):
            filepath = self.folder +'/' + filename
            if os.path.isfile(filepath) and filename[-4:]=='.txt':
                print(filepath)
                encoding = 'utf-8'
                with open(filepath, 'r', encoding=encoding, errors='surrogateescape') as f:
                    s = f.read()
                    s = s.replace('\r\n','\n')
                    s = s.replace('“', '"') # nltk tokenizer doesn't recognize these windows cp1252 characters
                    s = s.replace('”', '"')
                    text += s
        if debug:
            print('start of text:')
            print(text[:1000])
            print()

        print('Split into paragraphs, shuffle, recombine...') # ~1sec
        paragraphs = re.split(r"\n\n+", text)
        if debug: print('nparagraphs:',len(paragraphs))
        random.seed(seed)
        random.shuffle(paragraphs)
        text = '\n\n'.join(paragraphs)
        if debug:
            print('start of text after shuffling:')
            print(text[:1000])
            print()
        del paragraphs

        print('Tokenizing text (~15sec)...')
        tokens = tokenize.word_tokenize(text.lower())
        if debug:
            print('ntokens:',len(tokens))
            print('first tokens:',tokens[:100])

        print('Find vocabulary words...') # ~1sec
        token_freqs = nltk.FreqDist(tokens)
        token_counts = token_freqs.most_common(self.nvocab-1)
        index_to_token = [token_count[0] for token_count in token_counts]
        index_to_token.insert(0, '') # oov/unknown at position 0
        token_to_index = dict([(token,i) for i,token in enumerate(index_to_token)])
        if debug: print('start of index_to_token:',index_to_token[:10])

        print('Convert text to numeric sequence, skipping OOV words...') # ~1sec
        self.sequence = []
        for token in tokens:
            itoken = token_to_index.get(token)
            if itoken:
                self.sequence.append(itoken)
        nelements = len(self.sequence)
        self.sequence = np.array(self.sequence, dtype=np.int)
        if debug:
            print('nelements:',nelements)
            print('start of sequence:')
            print(self.sequence[:100])

        #.
        self.word_to_iword = token_to_index
        self.iword_to_word = {iword:word for iword,word in enumerate(index_to_token)}

        if debug:
            print('unique tokens in tokenized text:', len(self.word_to_iword)) # eg 190,000
            print('iword "the":', self.word_to_iword['the'])
            print('iword ".":',self.word_to_iword['.'])
            print('word 1:',self.iword_to_word[1])

            print('most common words:')
            for i in range(1,10):
                print(i,self.iword_to_word[i])

            print('least common words:')
            nunique = len(self.word_to_iword)
            for i in range(nunique-1, nunique-10, -1):
                print(i,self.iword_to_word[i])

            words = sorted(list(self.word_to_iword.keys()))
            print('first words in dictionary',words[:50])
            print('sample words in dictionary',random.sample(words,50))
            del words


    def split(self, n, nvalidate, ntest, train_amount=1.0, debug=False):
        """
        Split sequence into train, validate, test sets.
        n - size of subsequences
        nvalidate - number of validation sequences
        ntest - number of test sequences
        train_amount - percentage of the total training sequence to use
        """

        nelements = len(self.sequence)
        ntrain_total = nelements - nvalidate - ntest
        if ntrain_total<0: ntrain_total = nelements # for debugging cases
        ntrain = int(ntrain_total * train_amount)

        if debug:
            print('total training tokens available:',ntrain_total)
            print('ntraining tokens that will be used:',ntrain)
            print('nvalidation tokens:', nvalidate)
            print('ntest tokens:', ntest)

        def create_dataset(sequence, n, noffset, nelements):
            """
            Convert a sequence of values into an x,y dataset.
            sequence - sequence of integers representing words.
            noffset - starting point
            nelements - how much of the sequence to process
            ncontext - size of subsequences
            e.g. create_dataset([0,1,2,3,4,5,6,7,8,9], 2, 6, 3) =>
                 ([[2 3 4],[3 4 5],[4 5 6]], [5 6 7])
            """
            ncontext = n-1
            xs, ys = [], []
            noffset = max(0, noffset) # for debugging cases
            nelements = min(nelements, len(sequence)) # ditto
            for i in range(noffset, noffset + nelements - ncontext):
                x = sequence[i:i+ncontext]
                y = sequence[i+ncontext]
                xs.append(x)
                ys.append(y)
            x_set = np.array(xs)
            y_set = np.array(ys)
            return x_set, y_set

        print('Create train, validate, test sets...') # ~5sec
        x_train, y_train = create_dataset(self.sequence, n=n, noffset=0, nelements=ntrain)
        x_validate, y_validate = create_dataset(self.sequence, n=n, noffset=-ntest-nvalidate, nelements=nvalidate)
        x_test, y_test = create_dataset(self.sequence, n=n, noffset=-ntest, nelements=ntest)

        if debug:
            print('train data size:',len(x_train))
            print('validation data size:',len(x_validate)) # nvalidate - (n-1)
            print('test data size:',len(x_test)) # ditto
            print('x_train sample:')
            print(x_train[:5])
            print('y_train sample:')
            print(y_train[:5])

        return x_train, y_train, x_validate, y_validate, x_test, y_test



if __name__ == '__main__':

    # data = Data('gutenbergs')
    data = Data('alice1')
    data.prepare(nvocab=100, debug=0)
    n = 4
    #. just return train, test - let model handle validation split
    x_train, y_train, x_validate, y_validate, x_test, y_test = data.split(n=n, nvalidate=10000, ntest=10000, debug=1)
    print(x_train[:5])




