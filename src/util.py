
"""
Utility functions used by more than one module.
"""

import heapq
from datetime import datetime
import sys
import os
import re
import random

import numpy as np
from tabulate import tabulate



def create_dataset(sequence, n, noffset, nelements):
    """
    Convert a sequence of values into an x,y dataset.
    sequence - sequence of integers
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


def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    """
    Print strings with different encodings.
    Need this because some text has odd encodings and causes print to fail.
    source: http://stackoverflow.com/a/29988426/243392
    """
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)


def get_best_iword_probs(probs, k):
    """
    Return the best k words and normalized probabilities from the given probabilities.
    e.g. get_best_iword_probs([[0.1,0.2,0.3,0.4]], 2) => [(3,0.57),(2,0.43)]
    """
    iword_probs = [(iword,prob) for iword,prob in enumerate(probs[0])]
    # convert list to a heap, find k largest values
    best_pairs = heapq.nlargest(k, iword_probs, key=lambda pair: pair[1])
    # normalize probabilities
    total = sum([prob for iword,prob in best_pairs])
    if total!=0:
        best_iword_probs = [(iword,prob/total) for iword,prob in best_pairs]
    else:
        best_iword_probs = [(iword,0.0) for iword,prob in best_pairs]
    return best_iword_probs
# test
# probs = np.array([[0.1,0.2,0.3,0.4]])
# iword_probs = get_best_iword_probs(probs, 2)
# print(iword_probs)


def choose_iwords(iword_probs, k):
    """
    Choose k words at random weighted by probabilities.
    eg choose_iwords([(3,0.5),(2,0.3),(9,0.2)], 2) => [3,9]
    """
    iwords_all = [iword for iword,prob in iword_probs]
    probs = [prob for iword,prob in iword_probs]
    #. could choose without replacement here
    iwords = np.random.choice(iwords_all, k, probs) # weighted choice
    return iwords
# test
# print(choose_iwords([(3,0.5),(2,0.3),(9,0.2)], 2))


def generate_text(model, data, n, nwords=20, k=3):
    """
    Generate text from the given model with semi stochastic search.
    k - higher value increases the range of possible next word choices
    """
    x = np.zeros((1,n-1), dtype=int)
    # iword = 0
    iword = random.randint(1, data.nvocab)
    words = []
    for i in range(nwords):
        x = np.roll(x,-1) # flattens array, rotates to left, and reshapes it
        # print(x)
        if len(x[0])>0: # for n=1, x[0] will always be empty
            x[0,-1] = iword # insert new word
        probs = model.predict_proba(x, verbose=0)
        iword_probs = get_best_iword_probs(probs, k)
        iwords = choose_iwords(iword_probs, 1) # choose randomly
        iword = iwords[0]
        try: # in case iword is out of bounds - eg for tiny vocabulary
            word = data.iword_to_word[iword]
            words.append(word)
        except:
            pass
    sentence = ' '.join(words)
    return sentence


def table(df):
    """
    Convert a pandas dataframe into a more readable org table string.
    """
    s = tabulate(df, headers=df.columns, showindex=False, tablefmt="orgtbl")
    return s


def remove_text(regexp, s, to_char=0):
    """
    Remove text matching regular expression from given string.
    """
    match = re.search(regexp, s, re.MULTILINE)
    if match:
        start, end = match.span()
        if to_char==0:
            s = s[end:]
        elif to_char==-1:
            s = s[:start]
    return s


def filetitle(filepath):
    """
    Return the file title for the given path, eg "a/path/foo.txt" -> "foo".
    """
    filename = os.path.basename(filepath)
    filetitle = os.path.splitext(filename)[0]
    return filetitle


def mkdir(path):
    """
    Make a directory, ignoring errors
    """
    try:
        os.mkdir(path)
    except:
        pass


def softmax(x):
    """
    Calculate softmax values for given vector x.
    See http://stackoverflow.com/questions/34968722/softmax-function-python
    """
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


def get_best_tokens(d, k):
    """
    Return the best k tokens with their probabilities from the given dictionary.
    #. eg _____________
    """
    # convert list to a heap, find k largest values
    lst = list(d.items()) # eg [('a',5),('dog',1),...]
    best = heapq.nlargest(k, lst, key=lambda pair: pair[1])
    ntotal = sum(d.values())
    best_pct = [(k,v/ntotal) for k,v in best]
    return best_pct


if __name__=='__main__':

    s = """header
---start of thing---
contents
---end of thing---
footer
"""
    print(remove_text(r"^---start.*---", s, 0))
    print(remove_text(r"^---end", s, -1))


