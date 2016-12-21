
"""
Split a textfile by sentences into train, validate, test files.

Usage:
$ python src/split.py --ptrain 0.8 --pvalidate 0.1 --ptest 0.1 data/raw/all.txt data/split

"""


from __future__ import print_function, division

import sys
import os.path
from nltk import tokenize


def get_next_file(output_files, proportions):
    """
    Get next output file to write to based on specified proportions.

    output_files: a list of file handles

    proportions: a list of floating point numbers that add up to one,
      representing the proportion of text to be sent to each file.

    Returns a file handle.
    """
    # determine which file to write to by comparing the
    # current file size proportions against the given proportions,
    # writing to the first one found with a lower proportion than desired.
    nchars = [f.tell() for f in output_files] # file sizes
    ntotal = sum(nchars)
    if ntotal==0:
        return output_files[0]
    pcurrent = [n/ntotal for n in nchars] # file proportions
    # find file that needs more data
    for i in range(len(output_files)):
        if pcurrent[i] < proportions[i]:
            return output_files[i]
    return output_files[0]


def split(filename, output_folder, ptrain=0.8, pvalidate=0.1, ptest=0.1):
    """
    Split a textfile on sentences into train, validate, and test files.

    Will put resulting files in specified output folder with -train.txt etc
    appended.

    ptrain, pvalidate, ptest: proportion of original file to put into respective
      output files

    Note: need to split on sentences, not lines, otherwise would wind up with
    artificial word tuples.
    """

    assert abs(ptrain+pvalidate+ptest-1)<1e-6 # must add to 1.0

    # initialize
    # output_folder = os.path.dirname(filename) # eg 'data/split'
    suffixes = ('train','validate','test')
    proportions = (ptrain, pvalidate, ptest)
    filetitle = os.path.basename(filename)[:-4] # eg 'all'
    output_filenames = [output_folder + '/' + filetitle + '-' + suffix + '.txt' for suffix in suffixes]

    # open output files for writing
    try:
        os.mkdir(output_folder)
    except:
        pass
    output_files = []
    for output_filename in output_filenames:
        print(output_filename)
        f = open(output_filename, 'wb')
        output_files.append(f)

    # open source datafile (eg all.txt)
    f_data = open(filename, 'rb')

    # parse into sentences
    #. use generators for larger text somehow
    s = f_data.read()
    s = s.decode('utf-8') # gutenbergs are all utf-8
    sentences = tokenize.sent_tokenize(s)

    # walk over sentences, outputting to the different output files
    for sentence in sentences:
        sentence = sentence.replace('\r\n',' ')
        sentence = sentence.replace('\n',' ')
        print(sentence)
        print()
        f = get_next_file(output_files, proportions)
        f.write(sentence)
        f.write('\n\n')

    # close all files
    f_data.close()
    for f in output_files:
        f.close()


# command line handler
# see https://pypi.python.org/pypi/argh
import argh
argh.dispatch_command(split)




