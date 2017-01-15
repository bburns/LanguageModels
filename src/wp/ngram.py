
"""
n-gram word prediction model

Basic version - no backoff or smoothing.
"""

import os
import random
from pprint import pprint, pformat

import nltk
from nltk import tokenize
import tabulate

import model
import util
from benchmark import benchmark


class Ngram(model.Model):
    """
    n-gram model - initialize with n.
    For load, save, test methods, see model.py

    Stores a sparse multidimensional array of token counts.
    The sparse array is implemented as a dict of dicts.
    """

    def __init__(self, data, n=3, k=3, train_amount=1.0, name_includes=[]):
        """
        Create an n-gram model.
        data          - source of training and testing data
        train_amount  - percent or number of training characters to use
        name_includes - list of properties to include in model name
        """
        self.data = data # lightweight interface for data files
        self.n = n  # the n in n-gram
        self.k = k
        self.train_amount = train_amount
        self.name = "ngram (n=%d)" % n
        if name_includes:
            self.name += '-' + '-'.join([key+'-'+str(self.__dict__[key]) for key in name_includes]) # eg 'train_amount-1000'
        self.filename = "%s/ngram-(n-%d-train_amount-%s).pickle" % (data.model_folder, n, str(train_amount))
        self._d = {} # dictionary of dictionary of ... of counts
        self.trained = False
        self.load_time = None
        self.save_time = None
        self.train_time = None
        self.test_time = None
        print("Create model " + self.name)

    def train(self, force_training=False):
        """
        Train the model and save it, or load from file if available.
        """
        if force_training==False and os.path.isfile(self.filename):
            self.load() # see model.py - will set self.load_time
        else:
            print("Training model " + self.name)
            with benchmark("Got training data"):
                tokens = self.data.tokens('train', self.train_amount)
                token_tuples = nltk.ngrams(tokens, self.n)
            with benchmark("Trained model") as b:
                # train by adding ngram counts to model
                for token_tuple in token_tuples:
                    self._increment_count(token_tuple)
            self.train_time = b.time
            self.trained = True
            # save the model
            self.save()

    def _increment_count(self, token_tuple):
        """
        Increment the value of the multidimensional array at given index (token_tuple) by 1.
        """
        ntokens = len(token_tuple)
        d = self._d
        # need to iterate down the token stream to find the last dictionary,
        # where you can increment the counter.
        for i, token in enumerate(token_tuple):
            if i==ntokens-1: # at last dictionary
                if token in d:
                    d[token] += 1
                else:
                    d[token] = 1
            else:
                if token in d:
                    d = d[token]
                else:
                    d[token] = {}
                    d = d[token]

    def generate_token(self, tokens):
        """
        Get a random token following the given sequence.
        """
        if self.n==1:
            tokens = [] # no context - will just return a random token from vocabulary
        else:
            tokens = tokens[-self.n+1:] # an n-gram can only see the last n tokens
        # get the final dictionary, which contains the subsequent tokens and their counts
        d = self._d
        for token in tokens:
            if token in d:
                d = d[token]
            else:
                return None
        # pick a random token according to the distribution of subsequent tokens
        ntotal = sum(d.values()) # total occurrences of subsequent tokens
        p = random.random() # a random value 0.0-1.0
        stopat = p * ntotal # we'll get the cumulative sum and stop when we get here
        ntotal = 0
        for token in d.keys():
            ntotal += d[token]
            if stopat < ntotal:
                return token
        return d.keys()[-1] # right? #. test

    # def test(self, k=3, test_amount=1.0): #. use all data by default?
    #     """
    #     Test the model and return the accuracy score.
    #     """
    #     # get the test tokens
    #     tokens = self.data.tokens('test', test_amount)
    #     # print(tokens)
    #     ntokens = len(tokens)
    #     # run test on the models
    #     nright = 0
    #     with benchmark("Test model " + self.name) as b:
    #         for i in range(ntokens-self.n):
    #             prompt = tokens[i:i+self.n-1]
    #             actual = tokens[i+self.n-1]
    #             token_probs = self.predict(prompt, k) # eg [('barked',0.031),('slept',0.025)...]
    #             #. add selection to samples
    #             print('prompt',prompt,'actual',actual,'token_probs',token_probs)
    #             if token_probs: # can be None
    #                 predicted_tokens = [token_prob[0] for token_prob in token_probs]
    #                 if actual in predicted_tokens:
    #                     nright += 1
    #         npredictions = i + 1
    #         accuracy = nright / npredictions
    #     self.test_time = b.time
    #     self.test_score = accuracy
    #     return accuracy

    def generate(self):
        """
        Generate sentence of random text.
        """
        start1 = 'END' #. magic
        output = []
        input = [start1]
        if self.n>=3:
            start2 = random.choice(list(self._d[start1].keys()))
            input.append(start2)
            output.append(start2)
        if self.n>=4:
            start3 = random.choice(list(self._d[start1][start2].keys()))
            input.append(start3)
            output.append(start3)
        while True:
            next = self.generate_token(input)
            input.pop(0)
            input.append(next)
            output.append(next)
            if next=='END': #. magic
                break
        sentence = ' '.join(output)
        return sentence

    # def predict(self, tokens):
    def predict(self, prompt):
        """
        Get the most likely next k tokens following the given string.
        """
        #. use Vocab class?
        s = prompt.lower()
        tokens = prompt.split()
        #. add assert len(tokens)==self.n, or ignore too much/not enough info?
        # get the last dictionary, which contains the subsequent tokens and their counts
        d = self._d
        for token in tokens:
            if token in d:
                d = d[token]
            else:
                return None
        # find the most likely subsequent token
        # see http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        # maxtoken = max(d, key=d.get)
        # return maxtoken
        best_tokens = util.get_best_tokens(d, self.k)
        return best_tokens

    def __str__(self):
        """
        Return string representation of model.
        """
        propnames = "name data n train_amount trained train_time \
                     load_time save_time test_time test_score filename".split()
        rows = []
        for propname in propnames:
            propvalue = self.__dict__[propname]
            row = [propname, propvalue]
            rows.append(row)
        s = tabulate.tabulate(rows, ['Property', 'Value'])
        # s = str(rows)
        # s =
        # s = self.name + '\n'
        # s +=
        # self.data = data # lightweight interface for data files
        # self.n = n  # the n in n-gram
        # self.train_amount = train_amount
        # self._d = {} # dictionary of dictionary of ... of counts
        # self.trained = False
        # self.name = "ngram (n=%d)" % n
        # self.filename = "%s/ngram-(n-%d-amount-%.4f).pickle" % (data.model_folder, n, train_amount)
        # print("Create model " + self.name)
        return s



if __name__ == '__main__':

    # unknown_token = "UNKNOWN"
    # sentence_end_token = "END"
    # s = "The dog barked. The cat meowed. The dog ran away. The cat slept."
    # print(s)
    # # nvocab = 10
    # # split text into sentences
    # sentences = nltk.sent_tokenize(s)
    # print(sentences)
    # # append END tokens
    # sentences = ["%s %s" % (sent, sentence_end_token) for sent in sentences]
    # print(sentences)
    # # Tokenize the sentences into words
    # tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # print(tokenized_sentences)
    # # Replace all words not in our vocabulary with the unknown token
    # print('replace unknown words with UNKNOWN token')
    # for i, sent in enumerate(tokenized_sentences):
    #     tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    # print(tokenized_sentences)
    # # Create the training data
    # print('Create training data:')
    # X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    # y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    # # print('X_train:',X_train[500])
    # # print('y_train:',y_train[500])
    # print('X_train:',X_train) # eg tokenized: The dog ran down the hill . END
    # print('y_train:',y_train) # eg tokenized: dog ran down the hill . END

    # --------------------------------------

    from data import Data
    # data = Data('animals')
    data = Data('gutenbergs')
    # print('train text:', data.text('train',1000).replace('\n\n',' '))
    # print('test text:', data.text('test',1000))

    from tabulate import tabulate

    for n in (1,2,3):
        # model = Ngram(data, n=n)
        model = Ngram(data, n=n, train_amount=6000)
        model.train()
        model.test(test_amount=2000)
        print('accuracy:', model.test_score)
        df = model.test_samples
        print(tabulate(model.test_samples, showindex=False, headers=df.columns))
        # print(df)
        # print(model._d)
        s = model.generate()
        print('generate:', s)
        print()



