
"""
Experiments - manage experiments on different models.
Load/train/save/test/time models with given data.
"""

from __future__ import print_function, division
import os
import os.path
from datetime import datetime

import pandas as pd

from benchmark import benchmark


class Experiment(object):
    """
    Run an experiment on a set of models across a set of parameters.
    """
    def __init__(self, model_specs, data, params):
        """
        Construct experiment
        """
        self.model_specs = model_specs
        self.data = data
        self.params = params
        assert(len(params)==1) # only handles one param at a time
        # self.data.prepare() # make sure the data is cleaned and split up

    def run(self):
        """
        Train and test the models with different parameters.
        """
        stime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('-'*80)
        print('Experiment', stime)
        print('-'*80)
        print()
        param_name = self.params.keys()[0]
        param_values = self.params[param_name]
        train_times = []
        test_times = []
        test_scores = []
        cols = []
        rows = []
        for param_value in param_values:
            print('Parameter:', param_name, param_value)
            print()
            row_name = param_name + '=' + str(param_value)
            rows.append(row_name)
            train_time_row = []
            test_time_row = []
            test_score_row = []
            cols = []
            for (model_class, model_params) in self.model_specs:
                model = model_class(self.data, **model_params)
                print(model.name)
                cols.append(model.name)
                #. should timing be handled in experiment with benchmark? or return them from fns? 
                # with benchmark("Train model") as b:
                #     model.train()
                # with benchmark("Test model") as b:
                #     accuracy = model.test()
                model.train() # loads/saves model as needed
                model.test() #. should this return score, or access it as with time? 
                print('Score', model.test_score)
                train_time_row.append(model.train_time)
                test_time_row.append(model.test_time)
                test_score_row.append(model.test_score)
                print()
            train_times.append(train_time_row)
            test_times.append(test_time_row)
            test_scores.append(test_score_row)
            print()
        # print(train_times)
        # print(test_times)
        # print(test_scores)
        #. these need to be pandas tables
        # self.train_times = train_times
        # self.test_times = test_times
        # self.test_scores = test_scores
        self.train_times = pd.DataFrame(train_times, index=rows, columns=cols)
        self.test_times = pd.DataFrame(test_times, index=rows, columns=cols)
        self.test_scores = pd.DataFrame(test_scores, index=rows, columns=cols)
        print('Train Times')
        print(self.train_times)
        print()
        print('Test Times')
        print(self.test_times)
        print()
        print('Test Scores')
        print(self.test_scores)
        print()
                

    # def train_models(self):
    #     """
    #     Train models on different amounts of data.
    #     """
    #     train_tokens = None
    #     # models = []
    #     # model_folder = self.data.model_folder
    #     for (model_class, model_params) in self.model_specs:
    #         model = model_class(**model_params)
    #         # if model.avail():
    #         # print("create model object")
    #         model = model_class(model_folder=model_folder, nchars=nchars, **model_params) # __init__ method
    #         model = model.load() # load model if available
    #         if not model.trained:
    #             # get sequence of training tokens if needed (slow)
    #             if not train_tokens:
    #                 print("get complete stream of training tokens, nchars=%d" % nchars)
    #                 train_tokens = data.tokens('train', nchars)
    #             print("train model",model.name)
    #             model.train(train_tokens)
    #             print("save model",model.name)
    #             model.save()
    #         models.append(model)
    #     return models


    # def test_model_table(model_table, data, ntest_chars=10000, npredictions_max=1000, k=3):
    #     """
    #     Test all models, returning results in a pandas dataframe.
    #     """
    #     # cols = ['nchars'] + [model.name for model in models]
    #     models = model_table[0]
    #     cols = ['nchars'] + [model.name for model in models[1:]]
    #     table = []
    #     for models in model_table:
    #         ntrain_chars = models[0]
    #         # scores = test_models(models, data, ntest_chars, npredictions_max, k)
    #         scores = test_models(models[1:], data, ntest_chars, npredictions_max, k)
    #         # print()
    #         row = [ntrain_chars] + scores
    #         table.append(row)
    #     # return as a transposed pandas df
    #     df = pd.DataFrame(table, columns=cols)
    #     # df = pd.DataFrame(table)
    #     df = df.transpose()
    #     nchars_list = [models[0] for models in model_table]
    #     df.columns = nchars_list
    #     df = df.drop('nchars',axis=0)
    #     return df

    # def test_models(models, data, ntest_chars=None, npredictions_max=1000, k=3):
    #     """
    #     Test the given models on nchars of the given data's test tokens.
    #     Returns list of accuracy scores for each model
    #     """
    #     # get the test tokens
    #     print('get complete stream of test tokens, nchars=%d' % ntest_chars)
    #     test_tokens = data.tokens('test', ntest_chars)
    #     ntokens = len(test_tokens)
    #     # run test on the models
    #     scores = []
    #     for model in models:
    #         n = model.n
    #         nright = 0
    #         for i in range(ntokens-n):
    #             prompt = test_tokens[i:i+n-1]
    #             actual = test_tokens[i+n]
    #             tokprobs = model.predict(prompt, k) # eg [('barked',0.031),('slept',0.025)...]
    #             if tokprobs: # can be None
    #                 predicted_tokens = [tokprob[0] for tokprob in tokprobs]
    #                 if actual in predicted_tokens:
    #                     nright += 1
    #             if i > npredictions_max: break
    #         npredictions = i
    #         accuracy = nright / npredictions
    #         print("%s: accuracy = nright/total = %d/%d = %f" % (model.name, nright, npredictions, accuracy))
    #         scores.append(accuracy)
    #     return scores


if __name__ == '__main__':

    from data import Data
    import ngram
    import rnn

    specs = [
        [ngram.Ngram, {'n':1}],
        [ngram.Ngram, {'n':2}],
        [ngram.Ngram, {'n':3}],
        [rnn.Rnn, {}],
    ]
    data = Data('animals')
    print('text',data.text())
    params = {'train_amount':[0.5, 1.0]}

    exper = Experiment(specs, data, params)
    exper.run()
    # print(exper.test_scores)

    # specs = [[rnn.Rnn, {'nvocab':100}] ]
    # params = {'nhidden':[5,10,20,100]}
    # exper = Experiment(specs, data, params)
    # exper.run()


