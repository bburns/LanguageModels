
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
    def __init__(self, model_specs, data, params, test_amount=1.0):
        """
        Construct an experiment.
        model_specs - a list of model classes and parameters to train and test
        data        - a data object (eg Data('gutenbergs'))
        params      - additional params for the models (eg {'train_amount':1000})
        test_amount - amount of test text to use (percent or nchars)
        """
        self.model_specs = model_specs
        self.data = data
        self.params = params
        self.test_amount = test_amount
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
            # row_name = param_name + '=' + str(param_value)
            # rows.append(row_name)
            rows.append(param_value)
            train_time_row = []
            test_time_row = []
            test_score_row = []
            cols = []
            for (model_class, model_params) in self.model_specs:
                params = model_params.copy()
                params[param_name] = param_value
                # model = model_class(self.data, **model_params)
                print(params)
                model = model_class(self.data, **params)
                print(model.name)
                cols.append(model.name)
                #. should timing be handled in experiment with benchmark? or return them from fns?
                # with benchmark("Train model") as b:
                #     model.train()
                # with benchmark("Test model") as b:
                #     accuracy = model.test()
                model.train() # loads/saves model as needed
                model.test(test_amount=self.test_amount) #. should this return score, or access it as with time?
                print('Score', model.test_score)
                train_time_row.append(model.train_time)
                test_time_row.append(model.test_time)
                test_score_row.append(model.test_score)
                print()
            train_times.append(train_time_row)
            test_times.append(test_time_row)
            test_scores.append(test_score_row)
            print()
        # make pandas tables
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


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # need to import the same way we load/save models in the notebook for pickle to work
    import sys; sys.path.append('../')
    import wp

    specs = [
        [wp.ngram.Ngram, {'n':1}],
        [wp.ngram.Ngram, {'n':2}],
        [wp.ngram.Ngram, {'n':3}],
        [wp.rnn.Rnn, {}],
    ]
    data = wp.data.Data('gutenbergs')
    params = {'train_amount':[1000,2000,5000,10000,20000,40000,80000]}

    exper = Experiment(specs, data, params, test_amount=1000)
    exper.run()
    exper.test_scores.plot()
    plt.suptitle('Model accuracy comparison')
    plt.xlabel('train_amount')
    plt.ylabel('accuracy')
    plt.show()

    # specs = [[rnn.Rnn, {'nvocab':100}] ]
    # params = {'nhidden':[5,10,20,100]}
    # exper = Experiment(specs, data, params)
    # exper.run()

