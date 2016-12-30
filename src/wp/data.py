
"""
Data module - wraps all data and handles processing.

Usage:
data = Data('gutenbergs')
data.clean()
data.merge()
data.split()

"""

from __future__ import print_function, division

import sys
import os.path
import random
import glob
import re
from pprint import pprint
from collections import defaultdict

import pandas as pd
import nltk
from nltk import tokenize

import util


class Data(object):
    """
    Wrapper around all data files, with splitter and tokenizers.
    """

    def __init__(self, dataset):
        """
        Create a data object - contains little to no state - most is in predefined files.
        """
        escape = '../../' # escape from the Experiment subfolder, where this is called from
        dataset_folder = escape + 'data/' + dataset + '/'
        self.dataset = dataset
        self.model_folder   = dataset_folder + 'models/'
        self.raw_folder     = dataset_folder + '1-raw/'
        self.cleaned_folder = dataset_folder + '2-cleaned/'
        self.merged_folder  = dataset_folder + '3-merged/'
        self.split_folder   = dataset_folder + '4-split/'
        self.raw_files      = self.raw_folder + '*.txt'
        self.cleaned_files  = self.cleaned_folder + '*.txt'
        self.merged_file    = self.merged_folder + 'all.txt'
        self.splitparts = ('train','validate','test')
        self.source_files = {
            'raw': self.raw_files,
            'merged': self.merged_file,
            'train': self.split_folder + 'all-train.txt',
            'validate': self.split_folder + 'all-validate.txt',
            'test': self.split_folder + 'all-test.txt',
        }

    def clean(self):
        """
        Clean raw files - remove Gutenberg header/footers, table of contents, nonascii chars.
        """
        util.mkdir(self.cleaned_folder)
        for infile in glob.glob(self.raw_files):
            _, filetitle = os.path.split(infile)
            outfile = self.cleaned_folder + filetitle
            if not os.path.isfile(outfile):
                print('cleaning %s to %s' % (infile, outfile))
                with open(infile, 'rb') as f_in:
                    s = f_in.read()
                    s = s.replace('\r\n','\n') # dos2unix
                    s = self.clean_header_footer(s)
                    s = self.clean_table_of_contents(s)
                    # strip out any non-ascii characters - nltk complains otherwise
                    #. need better way to handle this - eg convert to miserables, not misrables
                    #. use decode?
                    s = re.sub(r'[^\x00-\x7f]',r'', s)
                    with open(outfile, 'wb') as f_out:
                        f_out.write(s)
        print("The raw files have been cleaned.")

    def clean_header_footer(self, s):
        """
        Remove the Gutenberg header and footer/license from the given string.
        """
        match = re.search(r"\*\*\*[ ]*START.*\*\*\*", s)
        if match: pos = match.span()[1]; s = s[pos:]
        match = re.search(r"\*\*\*[ ]*END.*\*\*\*", s)
        if match: pos = match.span()[0]; s = s[:pos]
        return s

    def clean_table_of_contents(self, s):
        """
        Remove table of contents from specific texts.
        """
        match = re.search(r"List of Illustrations", s) # les miserables
        if match: pos = match.span()[0]; s = s[pos:]
        match = re.search(r"\[Sidenote\: _Down the Rabbit-Hole_\]", s) # alice
        if match: pos = match.span()[1]; s = s[pos:]
        match = re.search(r"PART ONE--The Old Buccaneer", s) # treasure island
        if match: pos = match.span()[1]; s = s[pos:]
        match = re.search(r"CANON ALBERIC'S SCRAP-BOOK", s) # mrjames1905
        if match: pos = match.span()[1]; s = s[pos:]
        match = re.search(r"I. THE RIVER BANK", s) # windinwillows
        if match: pos = match.span()[1]; s = s[pos:]
        return s

    def merge(self):
        """
        Merge the cleaned files into one file if not done yet.
        """
        util.mkdir(self.merged_folder)
        if os.path.isfile(self.merged_file):
            print("The cleaned files have already been merged.")
        else:
            with open(self.merged_file, 'wb') as f_all:
                # for filename in glob.glob(self.rawfiles):
                for filename in glob.glob(self.cleaned_files):
                    with open(filename, 'rb') as f:
                        s = f.read()
                        f_all.write(s)
            print("The cleaned files have been merged.")

    def split(self, ptrain=0.8, pvalidate=0.0, ptest=0.2):
        """
        Split a textfile on sentences into train, validate, and test files.
        Will put resulting files in specified output folder with -train.txt etc appended.
        ptrain, pvalidate, ptest: proportion of original file to put into respective output files
        Note: we need to split on sentences, not lines, otherwise would wind up with
        artificial word tuples.
        """
        assert abs(ptrain + pvalidate + ptest - 1) < 1e-6 # must add to 1.0
        # initialize
        util.mkdir(self.split_folder)
        proportions = (ptrain, pvalidate, ptest)
        filetitle = os.path.basename(self.merged_file)[:-4] # eg 'all'
        output_filenames = [self.split_folder + '/' + filetitle + '-' + splitpart
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
            os.mkdir(self.split_folder)
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

    def analyze(self):
        """
        Gather some statistics on the datafiles.
        """
        rows = []
        cols = ['Text','Characters','Words','Sentences']
        for filepath in glob.glob(self.cleaned_files):
            with open(filepath, 'rb') as f:
                s = f.read()
                filename = os.path.basename(filepath)
                filetitle = os.path.splitext(filename)[0]
                sentences = tokenize.sent_tokenize(s)
                tokens = set(tokenize.word_tokenize(s))
                row = [filetitle, len(s), len(tokens), len(sentences)]
                rows.append(row)
        df = pd.DataFrame(rows, columns=cols)
        return df

    def text(self, source='merged', nchars=None):
        """
        Return contents of a data source up to nchars.
        """
        #. use generators
        filename = self.source_files[source]
        with open(filename, 'rb') as f:
            s = f.read()
            if nchars: s = s[:nchars]
        return s

    def sentences(self, source='merged', nchars=None):
        """
        Parse a data source into sentences up to nchars and return in a list.
        """
        #. use generators
        s = self.text(source, nchars)
        s = s.replace('\r\n',' ')
        s = s.replace('\n',' ')
        sentences = tokenize.sent_tokenize(s)
        return sentences

    # def tokenized_sentences(self, source, nchars=None):
    #     """
    #     Parse a data source into tokenized sentences up to nchars, return in list.
    #     """
    #     sentences = self.sentences(source, nchars)
    #     tokenized_sentences = [tokenize.word_tokenize(sentence) for sentence in sentences]
    #     #. trim vocab here, ie pass nvocab=None, use UNKNOWN where needed?
    #     return tokenized_sentences

    # def indexed_sentences(self, source, vocab, nchars=None):
    #     """
    #     Parse a data source into indexed sentences up to nchars, return in list.
    #     """
    #     # word_to_index = {'The':1,'dog':2,'cat':3,'slept':4,'barked':5,'UNKNOWN':6}
    #     sentences = self.tokenized_sentences(source, nchars)
    #     indexed_sentences = [[vocab.word_to_index[word] for word in sentence] for sentence in sentences]
    #     # # replace all words not in vocabulary with UNKNOWN
    #     # tokens = [token if token in self.word_to_index else unknown_token for token in tokens]
    #     # # replace words with numbers
    #     # itokens = [self.word_to_index[token] for token in tokens]
    #     # X_train = itokens[:-1]
    #     # y_train = itokens[1:]
    #     return indexed_sentences

    def tokens(self, source='merged', nchars=None):
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

    # def indexed_tokens(self, source, vocab, nchars=None):
    #     """
    #     Parse a data source into tokens up to nchars and return indices according to vocabulary.
    #     """
    #     tokens = self.tokens(source, nchars)
    #     indexed_tokens = [vocab.word_to_index[token] for token in tokens]
    #     return indexed_tokens

    # def tuples(self, source, ntokens_per_tuple, nchars=None):
    #     """
    #     Parse a data source into tokens up to nchars and return as tuples.
    #     """
    #     #. use generators!
    #     tokens = self.tokens(source)
    #     tokenlists = [tokens[i:] for i in range(ntokens_per_tuple)]
    #     tuples = zip(*tokenlists) # eg [['the','dog'], ['dog','barked'], ...]
    #     return tuples


# class Vocab(object):
#     """
#     """
#     def __init__(self, tokens, nvocab):
#         """
#         """
#         self.nvocab = nvocab
#         unknown_token = "UNKNOWN"
#         word_freqs = nltk.FreqDist(tokens)
#         wordcounts = word_freqs.most_common(nvocab-1)
#         self.index_to_word = [wordcount[0] for wordcount in wordcounts]
#         self.index_to_word.append(unknown_token)
#         # self.word_to_index = dict([(word,i) for i,word in enumerate(self.index_to_word)])
#         self.word_to_index = defaultdict(lambda: unknown_token)
#         for i, word in enumerate(self.index_to_word):
#             self.word_to_index[word] = i



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

    data = Data('animals')
    data.clean()
    data.merge()
    data.split()

    print(data.analyze())
    print(data.text())
    print(data.sentences())
    print(data.tokens())
    print(data.text('train'))

    # s = "header\n*** START OF TEXT ***\n contents \n*** END OF TEXT ***\n license"
    # s = data.clean_header_footer(s)
    # print(s)

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


    # s = 'The dog barked. The cat slept.'
    # nvocab = 3
    # vocab = Vocab(s.split(), nvocab)
    # print(vocab)
    # print(vocab.index_to_word)
    # print(vocab.word_to_index)

