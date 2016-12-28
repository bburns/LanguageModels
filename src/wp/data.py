
"""
Data module - wraps all data and handles processing.

Usage:
data = Data()

"""

from __future__ import print_function, division

import sys
import os.path
import random
import glob
import re
from pprint import pprint
from collections import defaultdict

import nltk
from nltk import tokenize


class Data():
    """
    Wrapper around all data files, with splitter and tokenizers.
    """

    def __init__(self):
        """
        Create a data object - contains little to no state - most is in predefined files.
        """
        #. could pass some of these as parameters
        self.escape = '../../' # escape from the Experiment subfolder, where this is called from
        self.rawfiles    = self.escape + 'data/raw/*.txt'
        self.mergedfile  = self.escape + 'data/merged/all.txt'
        self.splitfolder = self.escape + 'data/split/'
        self.splitparts = ('train','validate','test')

        self.sourcefiles = {
            'raw': self.rawfiles,
            'merged': self.mergedfile,
            'train': self.splitfolder + 'all-train.txt',
            'validate': self.splitfolder + 'all-validate.txt',
            'test': self.splitfolder + 'all-test.txt',
        }


    def merge(self):
        """
        Merge the raw files into one file if not done yet.
        Converts files to plain ascii text.
        """
        if os.path.isfile(self.mergedfile):
            print("The raw files have already been merged.")
        else:
            with open(self.mergedfile, 'wb') as f_all:
                for filename in glob.glob(self.rawfiles):
                    with open(filename, 'rb') as f:
                        s = f.read()
                        # strip out any non-ascii characters - nltk complains otherwise
                        s = re.sub(r'[^\x00-\x7f]',r'', s)
                        # s = str(s) # convert to plain string
                        s = s.replace('\r\n','\n') # dos2unix
                        f_all.write(s)
            print("The raw files have been merged.")

    def split(self, ptrain=0.8, pvalidate=0.1, ptest=0.1):
        """
        Split a textfile on sentences into train, validate, and test files.
        Will put resulting files in specified output folder with -train.txt etc appended.
        ptrain, pvalidate, ptest: proportion of original file to put into respective output files
        Note: we need to split on sentences, not lines, otherwise would wind up with
        artificial word tuples.
        """
        assert abs(ptrain + pvalidate + ptest - 1) < 1e-6 # must add to 1.0
        # initialize
        proportions = (ptrain, pvalidate, ptest)
        filetitle = os.path.basename(self.mergedfile)[:-4] # eg 'all'
        output_filenames = [self.splitfolder + '/' + filetitle + '-' + splitpart
                            + '.txt' for splitpart in self.splitparts] # eg 'all-train.txt'
        # do the output files already exist?
        allexist = True
        for output_filename in output_filenames:
            if not os.path.isfile(output_filename):
                allexist = False
                break
        if allexist:
            print("The merged file has already been split.")
            return
        # open output files for writing
        try:
            os.mkdir(self.splitfolder)
        except:
            pass
        output_files = []
        for output_filename in output_filenames:
            f = open(output_filename, 'wb')
            output_files.append(f)
        # parse merged file into sentences
        sentences = self.sentences('merged')
        # walk over sentences, outputting to the different output files
        for sentence in sentences:
            f = self._get_next_file(output_files, proportions)
            f.write(sentence)
            f.write('\n\n')
        # close all files
        for f in output_files:
            f.close()
        print("The merged file has been split into train, validate, and test files.")

    def _get_next_file(self, output_files, proportions):
        """
        Get next output file to write to based on specified proportions.
        This is used by split method to split a file into train, validate, test files.
        output_files: a list of file handles
        proportions: a list of floating point numbers that add up to one,
          representing the proportion of text to be sent to each file.
        Returns a file handle.
        """
        # determine which file to write to by comparing the current file size
        # proportions against the given proportions, writing to the first one
        # found with a lower proportion than desired.
        nchars = [f.tell() for f in output_files] # file sizes
        ntotal = sum(nchars)
        if ntotal==0:
            return output_files[0] # start with first file
        pcurrent = [n/ntotal for n in nchars] # file proportions
        # find file that needs more data
        for i in range(len(output_files)):
            if pcurrent[i] < proportions[i]:
                return output_files[i] # return the first under-appreciated file
        return output_files[0] # otherwise just return the first file

    def text(self, source, nchars=None):
        """
        Return contents of a data source up to nchars.
        """
        #. use generators
        filename = self.sourcefiles[source]
        with open(filename, 'rb') as f:
            s = f.read()
            if nchars: s = s[:nchars]
        return s

    def find_vocabulary(self, source, nvocab):
        """
        Find most used words and generate indices.
        """
        tokens = self.tokens(source)
        vocab = Vocab(tokens, nvocab)
        return vocab

    def sentences(self, source, nchars=None):
        """
        Parse a data source into sentences up to nchars and return in a list.
        """
        #. use generators
        s = self.text(source, nchars)
        s = s.replace('\r\n',' ')
        s = s.replace('\n',' ')
        sentences = tokenize.sent_tokenize(s)
        return sentences

    def tokenized_sentences(self, source, nchars=None):
        """
        Parse a data source into tokenized sentences up to nchars, return in list.
        """
        sentences = self.sentences(source, nchars)
        tokenized_sentences = [tokenize.word_tokenize(sentence) for sentence in sentences]
        #. trim vocab here, ie pass nvocab=None, use UNKNOWN where needed?
        return tokenized_sentences

    def indexed_sentences(self, source, vocab, nchars=None):
        """
        Parse a data source into indexed sentences up to nchars, return in list.
        """
        # word_to_index = {'The':1,'dog':2,'cat':3,'slept':4,'barked':5,'UNKNOWN':6}
        sentences = self.tokenized_sentences(source, nchars)
        indexed_sentences = [[vocab.word_to_index[word] for word in sentence] for sentence in sentences]
        # # replace all words not in vocabulary with UNKNOWN
        # tokens = [token if token in self.word_to_index else unknown_token for token in tokens]
        # # replace words with numbers
        # itokens = [self.word_to_index[token] for token in tokens]
        # X_train = itokens[:-1]
        # y_train = itokens[1:]
        return indexed_sentences

    def tokens(self, source, nchars=None):
        """
        Parse a data source into tokens up to nchars and return in a list.
        """
        #. trim vocab here? ie use UNKNOWN where needed?
        #. use generators
        sentences = self.sentences(source, nchars)
        tokens = []
        for sentence in sentences:
            # sentence = sentence.lower() # reduces vocab space
            words = tokenize.word_tokenize(sentence)
            tokens.extend(words)
            tokens.append('END') # add an END token to every sentence
        return tokens

    def indexed_tokens(self, source, vocab, nchars=None):
        """
        Parse a data source into tokens up to nchars and return indices according to vocabulary.
        """
        tokens = self.tokens(source, nchars)
        indexed_tokens = [vocab.word_to_index[token] for token in tokens]
        return indexed_tokens

    def tuples(self, source, ntokens_per_tuple, nchars=None):
        """
        Parse a data source into tokens up to nchars and return as tuples.
        """
        #. use generators!
        tokens = self.tokens(source)
        tokenlists = [tokens[i:] for i in range(ntokens_per_tuple)]
        tuples = zip(*tokenlists) # eg [['the','dog'], ['dog','barked'], ...]
        return tuples


class Vocab(object):
    """
    """
    def __init__(self, tokens, nvocab):
        """
        """
        self.nvocab = nvocab
        unknown_token = "UNKNOWN"
        word_freqs = nltk.FreqDist(tokens)
        wordcounts = word_freqs.most_common(nvocab-1)
        self.index_to_word = [wordcount[0] for wordcount in wordcounts]
        self.index_to_word.append(unknown_token)
        # self.word_to_index = dict([(word,i) for i,word in enumerate(self.index_to_word)])
        self.word_to_index = defaultdict(lambda: unknown_token)
        for i, word in enumerate(self.index_to_word):
            self.word_to_index[word] = i



# Split a textfile by sentences into train, validate, test files,
# based on specified proportions.
# Usage:
# >>> import split
# >>> split.split('data/raw/all.txt', 'data/split', 0.8, 0.1, 0.1)
# or
# $ python src/split.py --ptrain 0.8 --pvalidate 0.1 --ptest 0.1 data/raw/all.txt data/split
# if __name__ == '__main__':
#     # command line handler
#     # see https://pypi.python.org/pypi/argh
#     import argh
#     argh.dispatch_command(split)

if __name__ == '__main__':

    data = Data()
    data.merge()
    data.split()

    # tokens = data.tokens('train', 300)
    # print('train:',tokens)
    # print()

    # tokens = data.tokens('test', 300)
    # print('test:',tokens)
    # print()

    # sentences = data.sentences('train', 300)
    # print('train:',sentences)
    # print()

    # tokenized_sentences = data.tokenized_sentences('train', 300)
    # print('train:',tokenized_sentences)
    # print()

    # indexed_sentences = data.indexed_sentences('train', 300)
    # print('train:',indexed_sentences)
    # print()


    s = 'The dog barked. The cat slept.'
    nvocab = 3
    vocab = Vocab(s.split(), nvocab)
    print(vocab)
    print(vocab.index_to_word)
    print(vocab.word_to_index)

