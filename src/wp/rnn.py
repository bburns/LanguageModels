
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
    For load, save, test methods, see model.py Model class.
    """

    def __init__(self, data, train_amount=1.0, n=3, nvocab=1000, nhidden=100, nepochs=10, bptt_truncate=4, name_includes=[]):
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

        # unsure about these...
        self.n = n #... for now - used in test(). yes i think that's what we want - n-1 is amount of context given to model.
        # self.bptt_truncate = bptt_truncate #. -> ntimestepsmax?
        self.bptt_truncate = n #. -> call it ntimestepsmax? keep separate from n?
        self.seqlength = 10 #. -> call it nelements_per_sequence? instead of chopping up by sentences we'll chop up into sequences of this length

        self.name = "RNN-" + '-'.join([key+'-'+str(self.__dict__[key]) for key in name_includes]) # eg 'RNN-nhidden-10'
        self.filename = '%s/rnn-(train_amount-%s-nvocab-%d-nhidden-%d-nepochs-%d).pickle' \
                         % (data.model_folder, str(train_amount), nvocab, nhidden, nepochs)
        self.trained = False
        self.load_time = None
        self.save_time = None
        self.train_time = None
        self.test_time = None
        self.unknown_token = "UNKNOWN" #. ok?
        self.end_token = "END" #.
        print("Create model " + self.name)

    def train(self, force_training=False):
        """
        Train the model and save it, or load from file if available.
        force_training - set True to retrain model (ie don't load from file)
        """
        if force_training==False and os.path.isfile(self.filename):
            self.load() # see model.py - will set self.load_time
        else:
            #. fix train_amount output
            print("Training model %s on %s percent/chars of training data..." % (self.name, str(self.train_amount)))
            # time the training session
            # with benchmark("Trained model " + self.name) as b:
            with benchmark("Prepared training data"):
                print("Getting training tokens")
                #. would like this to memoize these if not too much memory, or else pass in tokens and calc them in Experiment class
                tokens = self.data.tokens('train', self.train_amount) # eg ['a','b','.','END']
                print(tokens)
                # get most common words for vocabulary
                word_freqs = nltk.FreqDist(tokens)
                # print(word_freqs)
                wordcounts = word_freqs.most_common(self.nvocab-1)
                # print(wordcounts)
                self.index_to_word = [wordcount[0] for wordcount in wordcounts]
                self.index_to_word.append(self.unknown_token)
                self.index_to_word.sort() #. just using for abcd dataset
                print(self.index_to_word)
                self.word_to_index = dict([(word,i) for i,word in enumerate(self.index_to_word)])
                self.nvocab = len(self.index_to_word)
                # print(self.word_to_index)
                # replace words not in vocabulary with UNKNOWN
                # tokens = [token if token in self.word_to_index else unknown_token for token in tokens]
                tokens = [token if token in self.word_to_index else self.unknown_token for token in tokens]
                # replace words with numbers
                itokens = [self.word_to_index[token] for token in tokens]
                print(itokens)
                # go through text some number of tokens at a time
                # so chop x and y into sequences of seqlength tokens
                #. or rnd # tokens? his orig code fed sentences to rnn, but that loses intersentence context
                seqs = []
                seq = []
                for i, itoken in enumerate(itokens):
                    seq.append(itoken)
                    if len(seq) >= self.seqlength:
                        seqs.append(seq)
                        seq = []
                seqs.append(seq) # add leftovers
                # seqs will be a list of sequences for rnn to learn, eg [[0,1,2,3],...]
                print('seqs',seqs)
                X_train = [seq[:-1] for seq in seqs] # eg [[0,1,2],...]
                y_train = [seq[1:] for seq in seqs] # eg [[1,2,3],...]
                print('Xtrain',X_train)
                print('ytrain',y_train)
                # parameters for the network that we need to learn
                #. use gaussians with suggested parameters
                self.U = np.random.uniform(-1,1, (self.nhidden, self.nvocab))
                self.V = np.random.uniform(-1,1, (self.nvocab, self.nhidden))
                self.W = np.random.uniform(-1,1, (self.nhidden, self.nhidden))
            with benchmark("Gradient descent finished") as b:
                print("Starting gradient descent")
                # train model with stochastic gradient descent - learns U, V, W
                # see model.py for fn
                learning_rate = 1.0 #.. for abcd dataset
                losses = self.train_with_sgd(X_train, y_train,
                                             learning_rate=learning_rate,
                                             nepochs=self.nepochs,
                                             evaluate_loss_after=int(self.nepochs/10))
            self.train_time = b.time
            self.trained = True
            self.train_losses = losses
            # save the model
            self.save()

    def sgd_step(self, x_sequence, y_sequence, learning_rate):
        """
        Perform one step of stochastic gradient descent (SGD).
        Adjust parameters U, V, W based on backpropagation gradients.
        """
        # calculate the gradients of the loss L with respect to parameters
        dLdU, dLdV, dLdW = self.bptt(x_sequence, y_sequence)
        # adjust parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def bptt(self, x_sequence, y_sequence):
        """
        Backpropagation Through Time (bptt)
        Calculate derivatives
        x_sequence - a sequence of vocabulary indexes, eg [0,1,2]
        y_sequence - a sequence of vocabulary indexes, eg [1,2,3]
        """
        nelements = len(y_sequence)
        # perform forward propagation to get output and state matrices
        output, state = self.forward_propagation(x_sequence)
        # we'll accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_output = output
        delta_output[np.arange(nelements), y_sequence] -= 1.0 # ?
        # iterate through sequence backwards
        # for t in range(nelements,0,-1): # eg for nelements=3 -> 3,2,1
        for t in np.arange(nelements)[::-1]: # eg for nelements=3 -> 2,1,0
            dLdV += np.outer(delta_output[t], state[t].T) # ?
            # initial delta calculation
            delta_t = self.V.T.dot(delta_output[t]) * (1 - (state[t] ** 2))
            # backpropagation through time (for at most self.bptt_truncate steps)
            #. why truncate steps? should we leave that off?
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print("Backpropagation step t=%d bptt step=%d " % (t, bptt_step))
                dLdW += np.outer(delta_t, state[bptt_step-1])
                dLdU[:, x_sequence[bptt_step]] += delta_t
                # update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - state[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def forward_propagation(self, x_sequence):
        """
        Do forward propagation for sequence x_sequence and return all output values and hidden states.
        x_sequence - sequence of numbers, eg [0,1,2], referring to words in the vocabulary.
        returns output, state
        output - a matrix of
        state - a matrix of
        """
        #. calculate amt of memory needed, eg in state, output, U,V,W, etc,
        #. and rough number of operations, as fns of sequence length, nhidden, nvocab, etc

        nelements = len(x_sequence)
        # will save all hidden states because need them later
        state = np.zeros((nelements + 1, self.nhidden)) # add 1 for initial hidden state. an nelements+1 x nhidden matrix
        state[-1] = np.zeros(self.nhidden) # set initial state to 0's
        # will save the outputs at each time step - will need later
        output = np.zeros((nelements, self.nvocab)) # an (nelements x nvocab) matrix
        # iterate over sequence
        for i in range(nelements): # eg [0,1,2]

            # original unexpanded code:
            # note that we are indexing U by x_sequence[t] -
            # this is the same as multiplying U with a one-hot vector.
            # ie picks out a column from the matrix U.
            # state[t] = np.tanh(self.U[:,x_sequence[t]] + self.W.dot(state[t-1])) # note how t-1=-1 for t=0
            # output[t] = util.softmax(self.V.dot(state[t]))

            # expanded code:
            # get the i'th word in the sequence
            iword = x_sequence[i] # eg 0
            # translate iword into a one-hot vector, eg
            # ohv = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,...0] # vector with nvocab elements
            # ohv = np.zeros(nvocab)
            # ohv[iword] = 1
            # multiply matrix U by ohv -
            # U is a matrix with (nvocab x nhidden) entries (eg 1e4 x 100 -> 1e6 entries).
            # note: this translates the word to a lower dimensional space with nhidden dimensions (eg 100) -
            # these will be the learned word embeddings, which we could replace with word2vec vectors, (?)
            # which will cut down a lot on the number of parameters needed to learn.
            # so U is a matrix of word embeddings (?)
            # u = self.U.dot(ohv) # dot is matrix multiply!
            # which is all equivalent to
            u = self.U[:,iword] # u is a vector with nhidden elements - this is the word embedding
            # now bring in the previous state, with more weights W -
            # W is a matrix with (nhidden x nhidden) elements (eg 100x100 = 1e4 entries)
            # note how t-1=-1 for t=0
            prevstate = state[i-1] # prevstate is a vector with nhidden elements
            w = self.W.dot(prevstate) # ie w = W x prevstate - dot is MATRIX MULTIPLY!
            # now add those and calculate the new state's activation value with a nonlinear fn, tanh.
            #. replace with ReLU?
            # u + w is the word embedding + context (?)
            state[i] = np.tanh(u + w) # state[t] is a vector with nhidden elements
            # output[t] = util.softmax(self.V.dot(state[t]))
            # now convert the internal/hidden state to scores for each class -
            # V is matrix with (nhidden x nvocab) entries (eg 100x1e4 = 1e6 entries) - converts an embedded word+context to vocab scores.
            #. but also can be doing some other work than just translation, eh?
            v = self.V.dot(state[i]) # v is a vector with nvocab elements, representing SCORES for each classifier.
            # ie each vocab word has a classifier that gets to vote for if it thinks it should be next.
            # now convert the scores to PROBABILITIES that all add to 1 -
            output[i] = util.softmax(v) # output[t] is a vector with nvocab PROBABILITY entries - will be lots of very small numbers
            # note that we don't convert to one-hot encodings or get argmax here, as need the probabilities for gradient descent
        # we not only return the calculated outputs, but also the hidden states.
        # we will use them later to calculate the gradients.
        # output - softmax output over the vocabulary for EACH time step - an nelements x nvocab matrix
        # state  - state of hidden layer elements - an nelements x nhidden matrix
        return [output, state]

    def total_loss(self, x, y):
        """
        Return total value of loss function for all training examples.
        #.The loss function is the Cross Entropy between the
        #.we're using negative log probability, which is entropy, ie amount of information ?
        #.we want to minimize total loss, ie entropy ?
        x - list of sequences, eg [[0,1,2],...]
        y - list of labels - y[i][j] should follow from x[i][0]...x[i][j-1], eg [[1,2,3],...]
        """
        total_loss = 0.0
        nsequences = len(x)
        # iterate over sequences
        for i in range(nsequences):
            x_sequence = x[i] # a sequence of numbers, eg [0,1,2]
            y_i = y[i] # the actual outputs - a sequence of numbers, eg [1,2,3]
            output, _ = self.forward_propagation(x_sequence) # get predicted outputs and internal states (don't need state though)
            nelements = len(x_sequence) # number of elements in this training sentence / sequence
            # we only care about our prediction of the "correct" words
            # correct_word_predictions = output[np.arange(len(y_i)), y_i]
            correct_word_predictions = output[np.arange(nelements), y_i]
            # add to the loss based on how off we were
            # ie if the probability is close to one, then the log will be close to 0, so little to no loss -
            # if the probability is close to zero, then the log will be towards -infinity, so add large number to loss.
            # so we'll try to minimize the total loss, which means all of the predictions are as correct as we can get them.
            #. where is y[i]?
            total_loss += -1 * np.sum(np.log(correct_word_predictions))
        print('U (words embedded in nhidden-d space)')
        print(self.U)
        print('V (next word predictor)')
        print(self.V)
        print('W (previous state weights)')
        print(self.W)
        print('output')
        print(output)
        print('output max')
        for i in range(len(output)):
            row = output[i]
            mr = row.max()
            row = row/mr
            print(row)
        return total_loss

    def average_loss(self, x, y):
        """
        Return average value of loss function per training example.
        """
        total_loss = self.total_loss(x, y)
        nexamples = np.sum((len(y_i) for y_i in y))
        avg_loss = total_loss / nexamples
        return avg_loss

    # see model.py for test()

    def predict(self, tokens, k):
        """
        Get the k most likely next tokens following the given sequence.
        """
        # print(tokens)
        # print(len(self.word_to_index))
        iwords = [self._get_index(word) for word in tokens]
        # print(iwords)
        output, state = self.forward_propagation(iwords)
        if len(output)>0:
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
        else:
            return []

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
            i = self.word_to_index[self.unknown_token]
        return i

    def generate(self):
        """
        Generate a sentence of random text.
        """
        iunknown = self.word_to_index[self.unknown_token]
        iend = self.word_to_index[self.end_token]
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
        # etc
        return s

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
    import pandas as pd
    # from tabulate import tabulate
    from data import Data

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


    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    data = Data('abcd')
    model = Rnn(data, nvocab=6, nhidden=2, nepochs=40, train_amount=100)
    model.train(force_training=True)
    model.test(test_amount=100)
    print('accuracy',model.test_score)
    print(util.table(model.test_samples))
    print()


    # data = Data('animals')
    # # data = Data('gutenbergs')
    # model = Rnn(data, nvocab=1000, nhidden=10, nepochs=10, train_amount=10000)
    # model.train(force_training=True)
    # model.test(test_amount=10000)
    # print('accuracy',model.test_score)
    # print(util.table(model.test_samples))
    # print()

    # # plot losses by epoch of training
    # df = model.train_losses
    # df.plot(x='Epoch', y='Loss')
    # plt.style.use('ggplot') # nicer style
    # plt.grid()
    # plt.title('Loss by Training Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    # # generate sentences
    # print("Generate sentences")
    # nsentences = 5
    # for i in range(nsentences):
    #     s = model.generate()
    #     print(s)
    # print()


