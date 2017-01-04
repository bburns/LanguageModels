
"""
Recurrent neural network (RNN) model

The RNN is effectively a deep network with any number of hidden layers,
all with the same parameters.
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
from pprint import pprint, pformat
# from collections import defaultdict

import numpy as np
import nltk
from nltk import tokenize

import model
import util
from benchmark import benchmark


class Rnn(model.Model):
    """
    Recurrent neural network (RNN) model.
    For load, save, test methods, see model.py
    """

    def __init__(self, data, train_amount=1.0, nvocab=1000, nhidden=100, nepochs=10, bptt_truncate=4, name_includes=[]):
        """
        Create an RNN model
        data          - source of training and testing data
        train_amount  - percent or number of training characters to use
        nvocab        - max number of vocabulary words to learn
        nhidden       - number of units in the hidden layer
        nepochs       - number of times to run through training data
        bptt_truncate - backpropagate through time truncation
        name_includes - list of properties to include in model name, eg ['nhidden']
        """
        self.data = data
        self.train_amount = train_amount
        self.nvocab = nvocab
        self.nhidden = nhidden
        self.nepochs = nepochs
        self.bptt_truncate = bptt_truncate #. -> ntimestepsmax?
        self.n = 2 #... for now - used in test()
        self.name = "rnn-" + '-'.join([key+'-'+str(self.__dict__[key]) for key in name_includes]) # eg 'rnn-nhidden-10'
        self.filename = '%s/rnn-(train_amount-%s-nvocab-%d-nhidden-%d-nepochs-%d).pickle' \
                         % (data.model_folder, str(train_amount), nvocab, nhidden, nepochs)
        self.trained = False
        self.load_time = None
        self.save_time = None
        self.train_time = None
        self.test_time = None
        print("Create model " + self.name)

    def train(self, force_training=False):
        """
        Train the model and save it, or load from file if available.
        force_training - pass True to retrain model (ie don't load from file)
        # Pass False to force training (or just delete the model files).
        """
        if force_training==False and os.path.isfile(self.filename):
            self.load() # see model.py - will set self.load_time
        else:
            print("Training model %s on %s percent/chars of training data..." % (self.name, str(self.train_amount)))
            # time the training session
            with benchmark("Trained model " + self.name) as b:
                #. just go through text 10 tokens at a time
                seqlength = 10 #...
                unknown_token = "UNKNOWN" #.
                print("Getting training tokens")
                tokens = self.data.tokens('train', self.train_amount)
                # get most common words for vocabulary
                word_freqs = nltk.FreqDist(tokens)
                wordcounts = word_freqs.most_common(self.nvocab-1)
                self.index_to_word = [wordcount[0] for wordcount in wordcounts]
                self.index_to_word.append(unknown_token)
                self.word_to_index = dict([(word,i) for i,word in enumerate(self.index_to_word)])
                self.nvocab = len(self.index_to_word)
                # replace words not in vocabulary with UNKNOWN
                tokens = [token if token in self.word_to_index else unknown_token for token in tokens]
                # print(tokens)
                # replace words with numbers
                itokens = [self.word_to_index[token] for token in tokens]
                # print(itokens)
                # chop x and y into sequences of 10 tokens. or rnd # tokens?
                seqs = []
                seq = []
                for i, itoken in enumerate(itokens):
                    seq.append(itoken)
                    if len(seq) >= seqlength:
                        seqs.append(seq)
                        seq = []
                seqs.append(seq)
                # print(seqs)
                X_train = [seq[:-1] for seq in seqs]
                y_train = [seq[1:] for seq in seqs]
                # print(X_train)
                # print(y_train)
                # parameters for the network that we need to learn
                self.U = np.random.uniform(-1,1, (self.nhidden, self.nvocab))
                self.V = np.random.uniform(-1,1, (self.nvocab, self.nhidden))
                self.W = np.random.uniform(-1,1, (self.nhidden, self.nhidden))
                # train model with stochastic gradient descent - learns U, V, W
                # see model.py for fn
                print("Starting gradient descent")
                losses = self.train_with_sgd(X_train, y_train, nepochs=self.nepochs, evaluate_loss_after=int(self.nepochs/10))
            self.train_time = b.time
            self.trained = True
            # save the model
            self.save()
            #. save the losses info with the model?
            # self.train_losses = losses

    # see model.py for test()

    def predict(self, tokens, k):
        """
        Get the most likely next k tokens following the given sequence.
        """
        # print(tokens)
        # print(len(self.word_to_index))
        iwords = [self._get_index(word) for word in tokens]
        # print(iwords)
        output, state = self.forward_propagation(iwords)
        next_word_probs = output[-1]
        # print(next_word_probs[:20])
        pairs = [(iword,p) for iword,p in enumerate(next_word_probs)]
        # print(pairs[:20])
        best_iwords = heapq.nlargest(k, pairs, key=lambda pair: pair[1])
        # print(best_iwords)
        # print(self.nvocab)
        # print(self.nvocab)
        # print(len(self.index_to_word))
        # print(self.index_to_word)
        best_words = [(self.index_to_word[iword],p) for iword,p in best_iwords]
        return best_words

    def _get_index(self, word):
        """
        Convert word to integer representation.
        """
        # tried using a defaultdict to return UNKNOWN instead of a dict, but
        # pickle wouldn't save an object with a lambda - would require defining
        # a fn just to return UNKNOWN. so this'll do.
        try:
            i = self.word_to_index[word]
        except:
            i = self.word_to_index["UNKNOWN"]
        return i

    def generate(self):
        """
        Generate a sentence of random text.
        """
        unknown_token = "UNKNOWN"
        end_token = "END"
        iunknown = self.word_to_index[unknown_token]
        iend = self.word_to_index[end_token]
        # start with the END token
        iwords = [iend]
        # repeat until we get another END token
        while True:
            output, state = self.forward_propagation(iwords)
            next_word_probs = output[-1]
            iword = iunknown
            # don't sample UNKNOWN words
            while iword == iunknown:
                sample = np.random.multinomial(1, next_word_probs)
                iword = np.argmax(sample)
            iwords.append(iword)
            if iword == iend:
                break
        tokens = [self.index_to_word[iword] for iword in iwords[1:-1]]
        sentence = ' '.join(tokens)
        return sentence

    def __str__(self):
        """
        Return model as a string.
        """
        s = self.name
        #. etc
        return s

    def forward_propagation(self, x):
        """
        Do forward propagation for sequence x and return output values and hidden states.
        x should be a list of numbers, eg [2, 4, 5, 1], referring to words in the vocabulary.
        output is the softmax output over the vocabulary for each time step (ie ~ a one-hot matrix).
        sstate is the internal state of the hidden layer for each time step.
        """
        nsteps = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden state, which we set to 0
        state = np.zeros((nsteps + 1, self.nhidden))
        state[-1] = np.zeros(self.nhidden)
        # The outputs at each time step. Again, we save them for later.
        # output = np.zeros((nsteps, self.nvocab))
        output = np.zeros((nsteps, self.nvocab))
        # For each time step...
        for t in np.arange(nsteps):
            # Note that we are indexing U by x[nstep] -
            # this is the same as multiplying U with a one-hot vector.
            # ie picks out a column from the matrix U.
            state[t] = np.tanh(self.U[:,x[t]] + self.W.dot(state[t-1])) # note how t-1=-1 for t=0
            output[t] = util.softmax(self.V.dot(state[t]))
        # We not only return the calculated outputs, but also the hidden states.
        # We will use them later to calculate the gradients.
        return [output, state]

    def total_loss(self, x, y):
        """
        Return total value of loss function for all training examples (?).
        x is a list of sentences (sequence of numbers)
        y is a list of labels, ie y[i][j] should follow from x[i][0]...x[i][j-1]
        """
        total_loss = 0
        # For each sentence...
        for i in np.arange(len(y)):
            output, state = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = output[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            total_loss += -1 * np.sum(np.log(correct_word_predictions))
        return total_loss

    def average_loss(self, x, y):
        """
        Return average value of loss function per training example.
        """
        total_loss = self.total_loss(x, y)
        nexamples = np.sum((len(y_i) for y_i in y))
        avg_loss = total_loss / nexamples
        return avg_loss

    def bptt(self, x, y):
        """
        Backpropagation through time
        """
        nsteps = len(y)
        # perform forward propagation
        output, state = self.forward_propagation(x)
        # accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_output = output
        delta_output[np.arange(nsteps), y] -= 1.0
        # for each output backwards...
        for t in np.arange(nsteps)[::-1]:
            dLdV += np.outer(delta_output[t], state[t].T)
            # initial delta calculation
            delta_t = self.V.T.dot(delta_output[t]) * (1 - (state[t] ** 2))
            # backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print("Backpropagation step t=%d bptt step=%d " % (t, bptt_step))
                dLdW += np.outer(delta_t, state[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_t
                # update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - state[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def sgd_step(self, x, y, learning_rate):
        """
        Perform one step of stochastic gradient descent (SGD).
        Adjust parameters U, V, W based on backpropagation gradients.
        """
        # calculate the gradients of the loss L with respect to parameters
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # adjust parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    # def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    #     """
    #     Calculate the gradients using backpropagation. We want to check if these are correct.
    #     """
    #     bptt_gradients = self.bptt(x, y)
    #     # List of all parameters we want to check.
    #     model_parameters = ['U', 'V', 'W']
    #     # Gradient check for each parameter
    #     for pidx, pname in enumerate(model_parameters):
    #         # Get the actual parameter value from the mode, e.g. model.W
    #         parameter = operator.attrgetter(pname)(self)
    #         print("Performing gradient check for parameter %s with size %d." % \
    #               (pname, np.prod(parameter.shape)))
    #         # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
    #         it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
    #         while not it.finished:
    #             ix = it.multi_index
    #             # Save the original value so we can reset it later
    #             original_value = parameter[ix]
    #             # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
    #             parameter[ix] = original_value + h
    #             gradplus = self.total_loss([x],[y])
    #             parameter[ix] = original_value - h
    #             gradminus = self.total_loss([x],[y])
    #             estimated_gradient = (gradplus - gradminus)/(2*h)
    #             # Reset parameter to original value
    #             parameter[ix] = original_value
    #             # The gradient for this parameter calculated using bptt
    #             bptt_gradient = bptt_gradients[pidx][ix]
    #             # calculate The relative error: (|x - y|/(|x| + |y|))
    #             relative_error = np.abs(bptt_gradient - estimated_gradient) / \
    #                              (np.abs(bptt_gradient) + np.abs(estimated_gradient))
    #             # If the error is to large fail the gradient check
    #             if relative_error > error_threshold:
    #                 print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
    #                 print("+h Loss: %f" % gradplus)
    #                 print("-h Loss: %f" % gradminus)
    #                 print("Estimated_gradient: %f" % estimated_gradient)
    #                 print("Bptt gradient: %f" % bptt_gradient)
    #                 print("Relative Error: %f" % relative_error)
    #                 return
    #             it.iternext()
    #         print("Gradient check for parameter %s passed." % (pname))



if __name__=='__main__':

    import matplotlib.pyplot as plt

    # # Check loss calculations
    # # Limit to 1000 examples to save time
    # print("Expected Loss for random predictions: %f" % np.log(nvocab))
    # print("Actual loss: %f" % model.average_loss(X_train[:1000], y_train[:1000]))

    # # Check gradient calculations
    # # use a smaller vocabulary size for speed
    # grad_check_vocab_size = 100
    # np.random.seed(10)
    # model = RnnModel(nvocab=grad_check_vocab_size, nhidden=10, bptt_truncate=1000)
    # model.gradient_check([0,1,2,3], [1,2,3,4])

    # print('see how long one sgd step takes')
    # np.random.seed(0)
    # model = Rnn(nvocab=nvocab, nhidden=nhidden)
    # with benchmark("Time for one sgd step"):
    #     model.sgd_step(X_train[1], y_train[1], 0.005)




    from data import Data

    # data = Data('animals')
    # model = Rnn(data, nvocab=10, nhidden=4, nepochs=1000)
    data = Data('gutenbergs')

    # m = Rnn(data,train_amount=1000)
    # print(m.filename)
    # stop

    model = Rnn(data, nvocab=1000, nhidden=10, nepochs=10, train_amount=10000)
    # model.train()
    model.train(True)
    # accuracy = model.test()
    accuracy = model.test(test_amount=10000)
    print('accuracy',accuracy)
    print()

    # plot losses per epoch of training
    # plt.line(model.train_losses)
    # plt.show()

    # # Sample predictions
    # tokens = "the dog".split()
    # k = 2
    # sample = model.predict(tokens, k)
    # print(sample)
    # print()

    # generate sentences
    print("Generate sentences")
    nsentences = 10
    for i in range(nsentences):
        s = model.generate()
        print(s)
    print()


