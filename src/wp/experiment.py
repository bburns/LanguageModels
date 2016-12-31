
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
    """
    def __init__(self, model_specs, data, params):
        """
        """
        self.model_specs = model_specs
        self.data = data
        self.params = params
        assert(len(params)==1) # only handles one param at a time
        self.data.prepare() # make sure the data is cleaned and split up

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
        scores = []
        times = []
        for param_value in param_values:
            print('Parameter:', param_name, param_value)
            print()
            score_row = []
            time_row = []
            for (model_class, model_params) in self.model_specs:
                model = model_class(self.data, **model_params)
                print(model.name)
                with benchmark("Train model") as b:
                    model.train()
                # print(b)
                with benchmark("Test model") as b:
                    accuracy = model.test()
                print('Accuracy', accuracy)
                row.append(accuracy)
                print()
            rows.append(row)
            print()
        print(rows)
        self.results = rows
                

    # def test(self):
    #     """
    #     Test the models against held-out test data.
    #     """
    #     #. time
    #     param_name = self.params.keys()[0]
    #     param_values = self.params[param_name]
    #     for param_value in param_values:
    #         for (model_class, model_params) in self.model_specs:
    #             model = model_class(**model_params)
    #             model.load()
    #             model.test()

    # def __str__(self):
    #     """
    #     """
    #     return "pokpok"


    # # def init_model_table(model_specs, model_folder, data, nchars_list=[None]):
    # # def init_model_table(model_specs, data, nchars_list=[None]):
    # def init_model_table(model_specs, data, train_amounts=[1.0]):
    #     """
    #     Initialize models
    #     """
    #     model_table = []
    #     # model_folder = data.model_folder
    #     for nchars in nchars_list:
    #         print('ntraining_chars', nchars)
    #         # load/train/save model
    #         # models = init_models(model_specs, model_folder, data, nchars=nchars) # load/train models
    #         models = init_models(model_specs, data, nchars=nchars) # load/train models
    #         models = [nchars] + models
    #         model_table.append(models)
    #     return model_table

    # # def init_models(model_specs, model_folder, data, nchars=None):
    # # def init_models(model_specs, data, nchars=None):
    # # def train_models(specs, data, train_amounts=[1.0]):
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
        # [rnn.Rnn, {}],
    ]
    data = Data('animals')
    print('text',data.text())
    params = {'train_amount':[0.5, 1.0]}

    exper = Experiment(specs, data, params)
    exper.run()
    print(exper.results)

    # specs = [[rnn.Rnn, {'nvocab':100}] ]
    # params = {'nhidden':[5,10,20,100]}
    # exper = Experiment(specs, data, params)
    # exper.run()





