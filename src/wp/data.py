
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
from nltk import tokenize


class Data():
    """
    Wrapper around all data files, with splitter and tokenizers.
    """

    def __init__(self):
        """
        Create a data object - contains little to no state - most is in predefined files.
        """
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
        Return contents of a data source.
        """
        #. use generators
        filename = self.sourcefiles[source]
        with open(filename, 'rb') as f:
            s = f.read()
            if nchars: s = s[:nchars]
        return s

    def sentences(self, source, nchars=None):
        """
        Parse a data source into sentences and return in a list.
        """
        #. use generators
        # filename = self.sourcefiles[source]
        # with open(filename, 'rb') as f:
        #     s = f.read()
        #     if nchars: s = s[:nchars]
        #     # sentences = self.get_sentences(s)
        #     s = s.replace('\r\n',' ')
        #     s = s.replace('\n',' ')
        #     sentences = tokenize.sent_tokenize(s)
        s = self.text(source, nchars)
        s = s.replace('\r\n',' ')
        s = s.replace('\n',' ')
        sentences = tokenize.sent_tokenize(s)
        return sentences

    def tokens(self, source, nchars=None):
        """
        Parse a data source into tokens and return in a list.
        """
        #. use generators
        # filename = self.sourcefiles[source]
        # with open(filename, 'rb') as f:
        #     s = f.read()
        #     if nchars: s = s[:nchars]
        #     tokens = tokenize.word_tokenize(s)
        # return tokens
        sentences = self.sentences(source, nchars)
        tokens = []
        for sentence in sentences:
            sentence = sentence.lower()
            words = tokenize.word_tokenize(sentence)
            tokens.extend(words)
            tokens.append('END') # add an END token to every sentence
        return tokens

    def tuples(self, source, ntokens_per_tuple, nchars=None):
        """
        Parse a data source into tokens and return as tuples.
        """
        #. use generators!
        tokens = self.tokens(source)
        tokenlists = [tokens[i:] for i in range(ntokens_per_tuple)]
        tuples = zip(*tokenlists) # eg [['the','dog'], ['dog','barked'], ...]
        return tuples

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
    tokens = data.tokens('train', 300)
    print(tokens)



