
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
import cPickle as pickle # faster version of pickle
from pprint import pprint, pformat

import nltk
from nltk import tokenize

import util


class RnnModel(object):
    """
    Recurrent neural network (RNN) model
    """

    def __init__(self, modelfolder='.', nchars=None, nwords=1000, nhidden=100, bptt_truncate=4):
        """
        Create an RNN model
        modelfolder - default location for model files
        nchars      - number of training characters to use
        nvocab      - number of vocabulary words to learn
        nhidden     - number of units in the hidden layer
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

    #. just use the .trained value - change in ngram and analyze also
    def trained(self):
        """
        Has this model been trained yet?
        """
        return self.trained

    def get_random(self, tokens):
        """
        Get a random token following the given sequence.
        """
        pass

    #. better - make n sentences of random text
    def generate(self, k):
        """
        Generate k tokens of random text.
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

    #. move save/load to baseclass wp.Model

    def save(self, filename=None):
        """
        Save the model to the default or given filename.
        """
        if filename is None:
            filename = self.filename()
        try:
            folder = os.path.dirname(filename)
            os.mkdir(folder)
        except:
            pass
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename=None):
        """
        Load model from the given or default filename.
        """
        if filename is None:
            filename = self.filename()
        if os.path.isfile(filename):
            print("load model")
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                return model
        else:
            return self

    # ----------------------------

    def forward_propagation(self, x):
        """
        Do forward propagation for sequence x and return output values and hidden states. (?)
        """
        ntimesteps = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden state, which we set to 0
        s = np.zeros((ntimesteps + 1, self.nhidden))
        s[-1] = np.zeros(self.nhidden)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((ntimesteps, self.nvocab))
        # For each time step...
        for ntimestep in np.arange(ntimesteps):
            # Note that we are indexing U by x[ntimestep] -
            # this is the same as multiplying U with a one-hot vector.
            s[ntimestep] = np.tanh(self.U[:,x[ntimestep]] + self.W.dot(s[ntimestep-1]))
            o[ntimestep] = util.softmax(self.V.dot(s[ntimestep]))
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
        total_loss = self.calculate_total_loss(x, y)
        avg_loss = total_loss / nexamples
        return avg_loss

    # how do we calculate those gradients we mentioned above? In a traditional
    # Neural Network we do this through the backpropagation algorithm. In RNNs we
    # use a slightly modified version of the this algorithm called Backpropagation
    # Through Time (BPTT). Because the parameters are shared by all time steps in
    # the network, the gradient at each output depends not only on the calculations
    # of the current time step, but also the previous time steps. If you know
    # calculus, it really is just applying the chain rule.

    def bptt(self, x, y):
        """
        Backpropagation through time
        """
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        """
        Calculate the gradients using backpropagation. We want to check if these are correct.
        """
        bptt_gradients = self.bptt(x, y)
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
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
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
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW



