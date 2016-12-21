
"""
Split a textfile into train, validate, test files.

Need to split on sentences, not lines, otherwise would wind up with artificial word tuples.
"""


from __future__ import print_function, division

# import sys #; sys.path.append('src')
# print(sys.path)

import os.path
from nltk import tokenize


def which_file(output_files, proportions):
    """
    """
    # nchars = map(output_files, lambda f: f.tell())
    nchars = [f.tell() for f in output_files]
    ntotal = sum(nchars)
    if ntotal==0:
        return output_files[0]
    pcurrent = [n/ntotal for n in nchars]
    for i in range(len(output_files)):
        if pcurrent[i] < proportions[i]:
            return output_files[i]
    return output_files[0]


def split(filename, ptrain=0.8, pvalidate=0.1, ptest=0.1):
    """
    Split a textfile on sentences into train, validate, and test files.
    """

    assert abs(ptrain+pvalidate+ptest-1)<1e-6

    proportions = (ptrain, pvalidate, ptest)
    suffixes = ('train','validate','test')
    output_folder = 'data/split'
    try:
        os.mkdir(output_folder)
    except:
        pass
    filetitle = os.path.basename(filename)[:-4] # eg 'all'

    # open files for writing
    output_files = []
    for suffix in suffixes:
        output_filename = output_folder + '/' + filetitle + '_' + suffix + '.txt'
        print(output_filename)
        f = open(output_filename, 'wb')
        output_files.append(f)

    f_all = open(filename, 'rb')

    #. use generator for larger text somehow
    s = f_all.read()
    sentences = tokenize.sent_tokenize(s)
    # ifile = 0
    for sentence in sentences:
        sentence = sentence.replace('\r\n',' ')
        sentence = sentence.replace('\n',' ')
        print(sentence)
        print()

        # nchars = len(sentence)
        # ntotal += nchars

        # if nfilechars[ifile] / ntotal < proportions[i]:
        # nfilechars[ifile] += nchars

        # decide which file to write to
        # ifile = which_file(output_files, proportions)
        # f = output_files[ifile]
        # ifile += 1
        # if ifile >= len(output_files):
        #     ifile = 0
        # track filesizes and proportions
        f = which_file(output_files, proportions)
        f.write(sentence)
        f.write('\n\n')

    f_all.close()

    for f in output_files:
        f.close()

