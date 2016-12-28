
"""
Utility functions used by more than one module.
"""

import heapq
from datetime import datetime
import sys

import numpy as np


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


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepochs=100, evaluate_loss_after=5):
    """
    Train model with Stochastic Gradient Descent (SGD)
    model               - the model instance
    X_train             - the training data set
    y_train             - the training data labels
    learning_rate       - initial learning rate for SGD
    nepochs             - number of times to iterate through the complete dataset
    evaluate_loss_after - evaluate the loss after this many epochs
    We keep track of the losses so we can plot them later
    """
    losses = []
    nexamples_seen = 0
    for nepoch in range(nepochs):
        # optionally evaluate the loss
        if (nepoch % evaluate_loss_after == 0):
            loss = model.average_loss(X_train, y_train)
            losses.append((nexamples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after nexamples_seen=%d epoch=%d: %f" % (time, nexamples_seen, nepoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # for each training example... (ie each sentence in his formulation)
        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learning_rate) # take one sgd step
            nexamples_seen += 1
    return losses

