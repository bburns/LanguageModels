
"""
Recurrent neural network (RNN) model

The RNN is effectively a network with any number of hidden layers,
all with the same weights.
"""

from __future__ import print_function, division

import os
import os.path
import random
import heapq
import re
import itertools
import operator
import time
import cPickle as pickle # faster version of pickle
from pprint import pprint, pformat

import numpy as np
import nltk
from nltk import tokenize

import util


class RnnModel(object):
    """
    Recurrent neural network (RNN) model
    """

    def __init__(self, modelfolder='.', nchars=None, nvocab=1000, nhidden=100, bptt_truncate=4):
        """
        Create an RNN model
        modelfolder - default location for model files
        nchars      - number of training characters to use
        nvocab      - number of vocabulary words to learn
        nhidden     - number of units in the hidden layer
        bptt_truncate - backpropagate through time truncation
        """
        self.modelfolder = modelfolder
        self.name = 'rnn'
        self.nvocab = nvocab
        self.nhidden = nhidden
        self.bptt_truncate = bptt_truncate
        self.trained = False
        # parameters for the network that we need to learn
        self.U = np.random.uniform(-1,1, (nhidden, nvocab))
        self.V = np.random.uniform(-1,1, (nvocab, nhidden))
        self.W = np.random.uniform(-1,1, (nhidden, nhidden))

    def filename(self):
        """
        Get default filename for model.
        """
        classname = type(self).__name__ # ie 'RnnModel'
        params = (('nchars',self.nchars),('nvocab',self.nvocab),('nhidden',self.nhidden))
        sparams = util.encode_params(params) # eg 'nchars-1000'
        filename = "%s/%s-%s.pickle" % (self.modelfolder, classname, sparams)
        return filename

    def train(self, tokens):
        """
        Train the rnn model with the given tokens.
        """
        # print("get ngrams, n=%d" % self.n)
        # token_tuples = nltk.ngrams(tokens, self.n)
        # print("add ngrams to model")
        # for token_tuple in token_tuples:
        #     self.increment(token_tuple)
        self.trained = True

    def get_random(self, tokens):
        """
        Get a random token following the given sequence.
        """
        pass

    def generate(self, k):
        """
        Generate k sentences of random text.
        """
        pass

    def predict(self, tokens, k):
        """
        Get the most likely next k tokens following the given sequence.
        """
        pass
    def predict(self, x):
        """
        Perform forward propagation and return index of highest score
        """
        o, s = self.forward_propagation(x)
        #. get k highest o values and their indices, translate to vocab words
        itoken = np.argmax(o, axis=1)
        return itoken

    def __str__(self):
        """
        Return model as a string.
        """
        pass

    # ----------------------------

    def forward_propagation(self, x):
        """
        Do forward propagation for sequence x and return output values and hidden states. (?)
        """
        nsteps = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        s = np.zeros((nsteps + 1, self.nhidden))
        # We added one additional element for the initial hidden state, which we set to 0
        s[-1] = np.zeros(self.nhidden)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((nsteps, self.nvocab))
        # For each time step...
        for nstep in np.arange(nsteps):
            # Note that we are indexing U by x[nstep] -
            # this is the same as multiplying U with a one-hot vector.
            s[nstep] = np.tanh(self.U[:,x[nstep]] + self.W.dot(s[nstep-1]))
            o[nstep] = util.softmax(self.V.dot(s[nstep]))
        # We not only return the calculated outputs, but also the hidden states.
        # We will use them later to calculate the gradients.
        return [o, s]

    #. should x be X?
    def total_loss(self, x, y):
        """
        Return total value of loss function for all training examples (?)
        """
        total_loss = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            total_loss += -1 * np.sum(np.log(correct_word_predictions))
        return total_loss

    def average_loss(self, x, y):
        """
        Return average value of loss function per training example (?)
        """
        # Divide the total loss by the number of training examples
        nexamples = np.sum((len(y_i) for y_i in y))
        total_loss = self.total_loss(x, y)
        avg_loss = total_loss / nexamples
        return avg_loss

    # how do we calculate those gradients we mentioned above? In a traditional
    # Neural Network we do this through the backpropagation algorithm. In RNNs we
    # use a slightly modified version of the this algorithm called Backpropagation
    # Through Time (BPTT). Because the parameters are shared by all time steps in
    # the network, the gradient at each output depends not only on the calculations
    # of the current time step, but also the previous time steps. If you know
    # calculus, it really is just applying the chain rule.

    def backpropagation(self, x, y):
        """
        Backpropagation through time
        """
        # T = len(y)
        nsteps = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        # delta_o[np.arange(len(y)), y] -= 1.0
        delta_o[np.arange(nsteps), y] -= 1.0
        # For each output backwards...
        # for t in np.arange(T)[::-1]:
        # for t in np.arange(nsteps)[::-1]:
        for nstep in np.arange(nsteps)[::-1]:
            # dLdV += np.outer(delta_o[t], s[t].T)
            dLdV += np.outer(delta_o[nstep], s[nstep].T)
            # Initial delta calculation
            # delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            delta_t = self.V.T.dot(delta_o[nstep]) * (1 - (s[nstep] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            # for backpropagation_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            for backpropagation_step in np.arange(max(0, nstep-self.bptt_truncate), nstep+1)[::-1]:
                # print "Backpropagation step t=%d backpropagation step=%d " % (t, backpropagation_step)
                dLdW += np.outer(delta_t, s[backpropagation_step-1])
                dLdU[:,x[backpropagation_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[backpropagation_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        """
        Calculate the gradients using backpropagation. We want to check if these are correct.
        """
        backpropagation_gradients = self.backpropagation(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = backpropagation_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / \
                                 (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))

    def sgd_step(self, x, y, learning_rate):
        """
        Perform one step of stochastic gradient descent (SGD).
        """
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.backpropagation(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW



if __name__=='__main__':

    import matplotlib.pyplot as plt

    nvocab = 100
    nhidden = 10
    infile = '../../data/raw/1865 Lewis Carroll Alice in Wonderland.txt'

    # strain = "the dog barked . END the cat meowed . END the dog ran away . END the cat slept ."
    # stest = "the cat"
    # # stest = "the cat slept"
    # train_tokens = strain.split()
    # test_tokens = stest.split()

    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "START_TOKEN"
    sentence_end_token = "END_TOKEN"

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading text...")
    with open(infile, 'rb') as f:
        # Split full comments into sentences
        s = f.read()
        s = re.sub(r'[^\x00-\x7f]',r'', s) # remove nonascii characters
        s = s.replace('\r\n','\n') # dos2unix
        s = s.replace('\n',' ')
        s = s.lower()
        sentences = nltk.sent_tokenize(s)
        # Append START and END tokens
        sentences = ["%s %s %s" % (sentence_start_token, sent, sentence_end_token) for sent in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(nvocab-1)
    print('vocab includes:',vocab[:20])
    index_to_word = [pair[0] for pair in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % nvocab)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % \
          (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print('Example sentence:')
    print(sentences[500])
    print('Tokenized:')
    print(tokenized_sentences[500])

    # Create the training data
    print('Create training data:')
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    print('X_train:',X_train[500]) # tokenized: The dog ran down the hill . END
    print('y_train:',y_train[500]) # tokenized: dog ran down the hill . END

    C = nvocab
    H = nhidden
    nparams = 2*H*C + H**2
    print("nparams to learn %d." % nparams)

    np.random.seed(0)
    model = RnnModel(nvocab=nvocab, nhidden=nhidden)
    o, s = model.forward_propagation(X_train[500])
    print('output matrix')
    print(o.shape)
    # print(o)

    print('predictions')
    predictions = model.predict(X_train[500])
    print(predictions.shape)
    print(predictions)

    # Limit to 1000 examples to save time
    print("Expected Loss for random predictions: %f" % np.log(nvocab))
    print("Actual loss: %f" % model.average_loss(X_train[:1000], y_train[:1000]))

    # # To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
    # grad_check_vocab_size = 100
    # np.random.seed(10)
    # model = RnnModel(grad_check_vocab_size, 10, bptt_truncate=1000)
    # model.gradient_check([0,1,2,3], [1,2,3,4])

    np.random.seed(0)
    model = RnnModel(nvocab=nvocab, nhidden=nhidden)
    t = time.time()
    model.sgd_step(X_train[500], y_train[500], 0.005)
    print('time for one sgd step: %f sec' % (time.time() - t))

    # Train on a small subset of the data to see what happens
    np.random.seed(0)
    model = RnnModel(nvocab=nvocab, nhidden=nhidden)
    losses = util.train_with_sgd(model, X_train[:100], y_train[:100], nepochs=10, evaluate_loss_after=1)
    print(losses)

    # plt.line(losses)
    # plt.show()





