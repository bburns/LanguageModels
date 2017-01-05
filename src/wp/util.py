
"""
Utility functions used by more than one module.
"""

from __future__ import print_function, division

import heapq
from datetime import datetime
import sys
import os
import re

import numpy as np
from tabulate import tabulate



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


