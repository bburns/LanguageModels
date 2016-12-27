
"""
Utility functions used by more than one module.
"""

import heapq


def softmax(x):
    """
    Calculate softmax values for given vector x.
    See http://stackoverflow.com/questions/34968722/softmax-function-python
    """
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def encode_params(params):
    """
    Encode a list of parameters as a string to be stored in a filename.
    e.g. (('n',3),('b',1.2)) => '(n-3-b-1.2)'
    """
    s = str(params)
    s = s.replace(",",'-')
    s = s.replace("'",'')
    s = s.replace('(','')
    s = s.replace(')','')
    s = s.replace(' ','')
    s = '(' + s + ')'
    return s

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

