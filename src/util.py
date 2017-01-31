
"""
Utility functions used by more than one module.
"""

import heapq
from datetime import datetime
import sys
import os
import re

import numpy as np
from tabulate import tabulate


def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    """
    Print strings with different encodings.
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
    best_iword_probs = [(iword,prob/total) for iword,prob in best_pairs]
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


#. make stochastic beam search
#. when have punctuation, start with period
#. stop when reach a period or max words
#. ->generate_sentence
#. k->beam_width
def generate_text(model, data, n, nwords=20, k=5):
    """
    Generate text from the given model with semi stochastic search.
    """
    x = np.zeros((1,n-1), dtype=int)
    iword = 0
    words = []
    for i in range(nwords):
        x = np.roll(x,-1) # flattens array, rotates to left, and reshapes it
        x[0,-1] = iword # insert new word
        probs = model.predict_proba(x, verbose=0)
        iword_probs = get_best_iword_probs(probs, k)
        iwords = choose_iwords(iword_probs, 1) # choose randomly
        iword = iwords[0]
        try:
            word = data.iword_to_word[iword]
            words.append(word)
        except:
            pass
    sentence = ' '.join(words)
    return sentence


# don't need - found top_k_categorical_accuracy
# #. a bit confusing - separate out fns? add example comments
# def get_relevance(actuals, probs, k):
#     """
#     Get relevance score for the given actual values and probabilities.
#     Checks if the actual value is included in the k most probable values,
#     for each row.
#     actuals - an array of one-hot encoded values, eg [[0 0 1], [0 1 0]]
#     probs   - an array of arrays of (value, probability) pairs
#     eg
#         actuals = np.array([[1,0,0,0],[0,1,0,0]]) # two one-hot encoded values
#         prob = [(0,0),(1,0.3),(2,0.3),(3,0.4)] # probabilities for 4 choices
#         probs = [prob, prob] # probabilities for 2 values
#         get_relevance(actual, probs) => 0.5
#     """
#     values = [row.argmax() for row in actuals] # get values, eg [2, 1]
#     nrelevant = 0
#     ntotal = 0
#     for value_probs, value in zip(probs, values):
#         pairs = [(value,prob) for (value,prob) in enumerate(value_probs)]
#         best_pairs = heapq.nlargest(k, pairs, key=lambda pair: pair[1])
#         best_values = [pair[0] for pair in best_pairs]
#         relevant = (value in best_values) # t or f
#         nrelevant += int(relevant) # 1 or 0
#         ntotal += 1
#     relevance = nrelevant/ntotal if ntotal!=0 else 0
#     return relevance
# # actuals = np.array([[1,0,0,0],[0,1,0,0]]) # two values
# # prob = [(0,0),(1,0.3),(2,0.3),(3,0.4)]
# # probs = [prob, prob]
# # relevance = get_relevance(actuals, probs)
# # print('relevance',relevance)
# # assert(relevance==0.5)
# # stop


# def cutoff(p):
#     """
#     Convert rows of probabilities to hard 0's and 1 (highest probability gets the 1).
#     eg cutoff(np.array([[0.1,0.1,0.8]])) => [[0. 0. 1.]]
#     """
#     onehot = np.zeros(p.shape)
#     for i in range(len(p)):
#         row = p[i]
#         mx = row.max()
#         row = row/mx
#         row = row.astype('int')
#         #. if all 1's, choose one at random to be 1, rest 0
#         onehot[i] = row
#     return onehot
# # print(cutoff(np.array([[0.1,0.1,0.8]])))
# # stop

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

