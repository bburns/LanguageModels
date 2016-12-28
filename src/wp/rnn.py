
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
        modelfolder   - default location for model files
        nchars        - number of training characters to use - None means use all
        nvocab        - number of vocabulary words to learn
        nhidden       - number of units in the hidden layer
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

    def train(self, tokens, nepochs=10):
        """
        Train the model with the given tokens.
        """
        #. could just go through text 10 tokens at a time?
        # print("get ngrams, n=%d" % self.n)
        # token_tuples = nltk.ngrams(tokens, self.n)
        # print("add ngrams to model")
        # for token_tuple in token_tuples:
        #     self.increment(token_tuple)
        unknown_token = "UNKNOWN"
        word_freqs = nltk.FreqDist(tokens)
        wordcounts = word_freqs.most_common(self.nvocab-1)
        self.index_to_word = [wordcount[0] for wordcount in wordcounts]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([(word,i) for i,word in enumerate(self.index_to_word)])
        # replace words not in vocabulary with UNKNOWN
        tokens = [token if token in self.word_to_index else unknown_token for token in tokens]
        print(tokens)
        # replace words with numbers
        itokens = [self.word_to_index[token] for token in tokens]
        print(itokens)
        # X_train = itokens[:-1]
        # y_train = itokens[1:]
        # split x and y into sequences of 10 tokens. or rnd # tokens.
        seqs = []
        seq = []
        for i, itoken in enumerate(itokens):
            seq.append(itoken)
            if len(seq)>=10:
                seqs.append(seq)
                seq = []
        seqs.append(seq)
        print(seqs)
        X_train = [seq[:-1] for seq in seqs]
        y_train = [seq[1:] for seq in seqs]
        print(X_train)
        print(y_train)
        # train model with stochastic gradient descent - learns U, V, W
        losses = util.train_with_sgd(self, X_train, y_train, nepochs=nepochs, evaluate_loss_after=int(nepochs/10))
        self.trained = True
        return losses

    def get_random(self, tokens):
        """
        Get a random token following the given sequence.
        """
        pass

    def generate(self):
        """
        Generate a sentence of random text.
        """
        unknown_token = "UNKNOWN"
        end_token = "END"
        iunknown = self.word_to_index[unknown_token]
        iend = self.word_to_index[end_token]
        # start the sentence with the END token
        iwords = [iend]
        # repeat until we get an end token
        while True:
            o, s = self.forward_propagation(iwords)
            next_word_probs = o[-1]
            iword = iunknown
            # don't want to sample unknown words
            while iword == iunknown:
                samples = np.random.multinomial(1, next_word_probs)
                iword = np.argmax(samples)
            iwords.append(iword)
            if iword == iend:
                break
        # s = [index_to_word[iword] for iword in iwords[1:-1]]
        s = [self.index_to_word[iword] for iword in iwords[1:-1]]
        return s

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
        s = self.name
        return s

    # ----------------------------

    def forward_propagation(self, x):
        """
        Do forward propagation for sequence x and return output values and hidden states.
        x should be a list of numbers, eg [2, 4, 5, 1], referring to words in the vocabulary.
        o is the softmax output over the vocabulary for each time step (ie ~ a one-hot matrix).
        s is the internal state of the hidden layer for each time step.
        """
        nsteps = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        s = np.zeros((nsteps + 1, self.nhidden))
        # We added one additional element for the initial hidden state, which we set to 0
        s[-1] = np.zeros(self.nhidden)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((nsteps, self.nvocab))
        # For each time step...
        for t in np.arange(nsteps):
            # Note that we are indexing U by x[nstep] -
            # this is the same as multiplying U with a one-hot vector.
            # ie picks out a column from the matrix U.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1])) # note how t-1=-1 for t=0
            o[t] = util.softmax(self.V.dot(s[t]))
        # We not only return the calculated outputs, but also the hidden states.
        # We will use them later to calculate the gradients.
        return [o, s]

    def total_loss(self, x, y):
        """
        Return total value of loss function for all training examples (?).
        x is a list of sentences (sequence of numbers)
        y is a list of labels, ie y[i][j] should follow from x[i][0]...x[i][j-1]
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

    def backpropagation(self, x, y):
        """
        Backpropagation through time
        """
        nsteps = len(y)
        # perform forward propagation
        o, s = self.forward_propagation(x)
        # accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(nsteps), y] -= 1.0
        # for each output backwards...
        for t in np.arange(nsteps)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # backpropagation through time (for at most self.bptt_truncate steps)
            for backpropagation_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print("Backpropagation step t=%d backpropagation step=%d " % (t, backpropagation_step))
                dLdW += np.outer(delta_t, s[backpropagation_step-1])
                dLdU[:,x[backpropagation_step]] += delta_t
                # update delta for next step
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
            print("Performing gradient check for parameter %s with size %d." % \
                  (pname, np.prod(parameter.shape)))
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
        Adjusts parameters U, V, W based on backpropagation gradients.
        """
        # calculate the gradients
        dLdU, dLdV, dLdW = self.backpropagation(x, y)
        # change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW



if __name__=='__main__':

    s = "The dog barked. The cat meowed. The dog ran away. The cat slept."
    print(s)

    nvocab = 10
    nhidden = 5

    unknown_token = "UNKNOWN"
    # end_token = "END"

    # split text into sentences
    sentences = nltk.sent_tokenize(s)
    print(sentences)

    # # append END tokens
    # sentences = ["%s %s" % (sent, sentence_end_token) for sent in sentences]
    # print(sentences)

    # # Tokenize the sentences into words
    # tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # print(tokenized_sentences)

    tokens = []
    for sentence in sentences:
        sentence = sentence.lower()
        words = nltk.word_tokenize(sentence)
        tokens.extend(words)
        tokens.append('END') # add an END token to every sentence

    # # Count the word frequencies
    # word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # print("Found %d unique words tokens." % len(word_freq.items()))

    # # Get the most common words and build index_to_word and word_to_index vectors
    # vocab = word_freq.most_common(nvocab-1)
    # print('most common words',vocab)
    # index_to_word = [pair[0] for pair in vocab]
    # index_to_word.append(unknown_token)
    # print('index to word',index_to_word)
    # word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    # print('word to index',word_to_index)

    # # print("Using vocabulary size %d." % nvocab)
    # # print("The least frequent word in our vocabulary is '%s' and appeared %d times." % \
    # #       (vocab[-1][0], vocab[-1][1]))

    # # Replace all words not in our vocabulary with the unknown token
    # print('replace unknown words with UNKNOWN token')
    # for i, sent in enumerate(tokenized_sentences):
    #     tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    # print(tokenized_sentences)

    # # print('Example sentence:')
    # # print(sentences[500])
    # # print('Tokenized:')
    # # print(tokenized_sentences[500])

    # # Create the training data
    # print('Create training data:')
    # X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    # y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    # # print('X_train:',X_train[500])
    # # print('y_train:',y_train[500])
    # print('X_train:',X_train) # eg tokenized: The dog ran down the hill . END
    # print('y_train:',y_train) # eg tokenized: dog ran down the hill . END


    C = nvocab
    H = nhidden
    nparams = 2*H*C + H**2
    print("nparams to learn %d." % nparams)
    print()

    # print('Train model on one sentence')
    # print('input matrix')
    # print(X_train[1])
    # s = [index_to_word[i] for i in X_train[1]]
    # print(s)
    # np.random.seed(0)
    # model = RnnModel(nvocab=nvocab, nhidden=nhidden)
    # o, s = model.forward_propagation(X_train[1])
    # print('state matrix (hidden unit state, per time step)')
    # print(s.shape)
    # print(s)

    # print('output matrix (softmax over vocabulary, per time step)')
    # print(o.shape)
    # print(o)

    # print('predictions')
    # predictions = model.predict(X_train[1])
    # print(predictions.shape)
    # print(predictions)
    # s = [index_to_word[i] for i in predictions]
    # print(s)
    # print('actual')
    # print(y_train[1])
    # s = [index_to_word[i] for i in y_train[1]]
    # print(s)

    # print()


    # # Check loss calculations
    # # Limit to 1000 examples to save time
    # print("Expected Loss for random predictions: %f" % np.log(nvocab))
    # print("Actual loss: %f" % model.average_loss(X_train[:1000], y_train[:1000]))

    # # Check gradient calculations
    # # To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
    # grad_check_vocab_size = 100
    # np.random.seed(10)
    # model = RnnModel(nvocab=grad_check_vocab_size, nhidden=10, bptt_truncate=1000)
    # model.gradient_check([0,1,2,3], [1,2,3,4])


    # print('see how long one sgd step takes')
    # np.random.seed(0)
    # model = RnnModel(nvocab=nvocab, nhidden=nhidden)
    # t = time.time()
    # # model.sgd_step(X_train[500], y_train[500], 0.005)
    # model.sgd_step(X_train[1], y_train[1], 0.005)
    # print('time for one sgd step: %f sec' % (time.time() - t))

    # Train on data
    print('Train model on data')
    np.random.seed(0)
    model = RnnModel(nvocab=nvocab, nhidden=nhidden)
    # nepochs = 2000
    nepochs = 100
    model.train(tokens, nepochs)
    # losses = util.train_with_sgd(model, X_train[:100], y_train[:100], nepochs=10, evaluate_loss_after=1)
    # losses = util.train_with_sgd(model, X_train, y_train, nepochs=10, evaluate_loss_after=1)
    # losses = util.train_with_sgd(model, X_train, y_train, nepochs=nepochs, evaluate_loss_after=int(nepochs/10))
    # print(losses)
    # print()

    # import matplotlib.pyplot as plt
    # plt.line(losses)
    # plt.show()

    # generate sentences
    print("Generate sentences")
    nsentences = 10
    nwordsmin = 2
    for i in range(nsentences):
        tokens = []
        while len(tokens) < nwordsmin:
            tokens = model.generate()
        print(' '.join(tokens))
